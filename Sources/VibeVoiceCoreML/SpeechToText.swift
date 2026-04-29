@preconcurrency import AVFoundation
import CoreML
import Foundation
import Tokenizers

// MARK: - Public API

/// Configuration for speech-to-text transcription.
public struct STTConfig: Sendable {
    /// Maximum number of tokens to generate. nil = auto-estimate from audio duration.
    public var maxNewTokens: Int?
    /// Optional prompt/context to guide transcription.
    public var prompt: String?

    public init(maxNewTokens: Int? = nil, prompt: String? = nil) {
        self.maxNewTokens = maxNewTokens
        self.prompt = prompt
    }
}

/// Result of a speech-to-text transcription.
public struct TranscriptionResult: Sendable {
    /// The transcribed text (JSON format with timestamps and speaker IDs).
    public let text: String
    /// Audio duration in seconds.
    public let audioDuration: Double
    /// Number of tokens generated.
    public let tokensGenerated: Int
    /// Generation time in seconds (excluding model load).
    public let generationTime: Double
}

// MARK: - ASR Constants

private enum ASRConstants {
    static let hiddenSize = 3584
    static let headDim = 128
    static let vocabSize = 152064

    static let sampleRate = 24000
    static let hopLength = 3200
    static let vaeDim = 64
    static let semDim = 128
    static let vaeStd: Float = 0.625

    static let chunkSeconds = 60
    static let chunkSamples = sampleRate * chunkSeconds  // 1,440,000
    static let chunkTokens = chunkSamples / hopLength    // 450

    // Special tokens
    static let audioTokenID = 151648    // placeholder for audio embeddings
    static let audioBosID = 151646      // start of audio region
    static let audioEosID = 151647      // end of audio region
    static let eosID = 151643           // generation stop
    static let imStartID = 151644
    static let imEndID = 151645

    static let ropeTheta: Double = 1e6
    static let maskValue: Float = -1e9
}

// MARK: - Speech-to-Text Engine

/// Speech-to-text using VibeVoice ASR 7B model.
public final class SpeechToText: @unchecked Sendable {
    private let fusedEncoder: MLModel
    private let fusedProjector: MLModel
    private let lmDecoder: MLModel
    private let embedTokens: EmbeddingTable
    private let tokenizerDir: URL

    // Pre-computed prompt token sequences
    private let promptPrefix: [Int]
    private let promptSuffix: [Int]
    private let nlToken: Int

    public init(modelsDir: URL, tokenizerDir: URL) throws {
        let configCPUGPU = MLModelConfiguration()
        configCPUGPU.computeUnits = .cpuAndGPU
        let configAll = MLModelConfiguration()

        func load(_ name: String, config cfg: MLModelConfiguration = configAll) throws -> MLModel {
            let compiledURL = modelsDir.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiledURL.path) {
                return try MLModel(contentsOf: compiledURL, configuration: cfg)
            }
            let packageURL = modelsDir.appendingPathComponent("\(name).mlpackage")
            let compiledDir = try MLModel.compileModel(at: packageURL)
            return try MLModel(contentsOf: compiledDir, configuration: cfg)
        }

        fusedEncoder = try load("fused_encoder", config: configCPUGPU)
        fusedProjector = try load("fused_projector")
        lmDecoder = try load("lm_decoder_fused_int8", config: configCPUGPU)
        embedTokens = try EmbeddingTable(
            contentsOf: modelsDir.appendingPathComponent("embed_tokens.bin")
        )
        self.tokenizerDir = tokenizerDir

        // Pre-compute prompt token sequences
        let tokenizer = try SpeechToText.loadTokenizerSync(from: tokenizerDir)

        let systemMsg = "You are a helpful assistant that transcribes audio input into text output in JSON format."
        let systemIds = tokenizer.encode(text: "system\n\(systemMsg)")
        let userPrefix = tokenizer.encode(text: "user\n")
        let assistantIds = tokenizer.encode(text: "assistant\n")
        let nl = tokenizer.encode(text: "\n")
        guard let nlTok = nl.first else {
            throw VibeVoiceError.modelError("Failed to encode newline token")
        }
        nlToken = nlTok

        promptPrefix = [ASRConstants.imStartID] + systemIds
            + [ASRConstants.imEndID, nlToken, ASRConstants.imStartID] + userPrefix
        promptSuffix = [ASRConstants.imStartID] + assistantIds
    }

    private static func loadTokenizerSync(from dir: URL) throws -> any Tokenizer {
        // TokenizerLoader is async, but we need sync for init
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: Result<any Tokenizer, Error>!
        Task {
            do {
                let tok = try await TokenizerLoader.load(from: dir)
                result = .success(tok)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try result.get()
    }

    /// Transcribe audio from a file URL.
    public func transcribe(_ audioURL: URL, config: STTConfig = STTConfig()) async throws -> TranscriptionResult {
        let hs = ASRConstants.hiddenSize
        let hd = ASRConstants.headDim

        // Load and preprocess audio
        let wav = try loadAudio(url: audioURL)
        let durationSecs = Double(wav.count) / Double(ASRConstants.sampleRate)
        let numAudioTokens = (wav.count + ASRConstants.hopLength - 1) / ASRConstants.hopLength

        // Estimate max tokens from audio duration: ~12 tok/s for JSON output with headroom
        let tokenLimit = config.maxNewTokens ?? max(256, Int(durationSecs * 12))

        let t0 = CFAbsoluteTimeGetCurrent()

        // Encode audio (chunked, 60s per chunk)
        let audioEmbeddings = try encodeAudio(wav: wav)

        // Build prompt
        let inputIds = buildPromptIds(
            numAudioTokens: numAudioTokens,
            durationSecs: durationSecs,
            prompt: config.prompt
        )

        // Build prefill embeddings
        var prefillEmbeds = embedTokens.lookup(inputIds)

        // Overwrite audio token positions with projected audio embeddings
        var audioIdx = 0
        for (i, id) in inputIds.enumerated() {
            if id == ASRConstants.audioTokenID && audioIdx < audioEmbeddings.count / hs {
                let srcOff = audioIdx * hs
                let dstOff = i * hs
                for j in 0 ..< hs {
                    prefillEmbeds[dstOff + j] = audioEmbeddings[srcOff + j]
                }
                audioIdx += 1
            }
        }

        // Prefill
        let seqLen = inputIds.count
        let state = lmDecoder.makeState()

        let positions = Array(0 ..< seqLen)
        let (cosArr, sinArr) = computeRoPE(positions: positions, headDim: hd)

        // Build causal mask
        let maskCount = seqLen * seqLen
        var maskData = [Float](repeating: 0, count: maskCount)
        for i in 0 ..< seqLen {
            for j in (i + 1) ..< seqLen {
                maskData[i * seqLen + j] = ASRConstants.maskValue
            }
        }

        let prefillInput = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": makeMultiArray(prefillEmbeds, shape: [1, seqLen, hs]),
            "position_cos": makeMultiArray(cosArr, shape: [1, seqLen, hd]),
            "position_sin": makeMultiArray(sinArr, shape: [1, seqLen, hd]),
            "attention_mask": makeMultiArray(maskData, shape: [1, 1, seqLen, seqLen]),
        ] as [String: MLMultiArray])

        let prefillOut = try await lmDecoder.prediction(from: prefillInput, using: state)
        guard let logits = prefillOut.featureValue(for: "logits")?.multiArrayValue else {
            throw VibeVoiceError.modelError("Missing logits from LM prefill")
        }
        var nextToken = argmaxMultiArray(logits, lastTokenOffset: (seqLen - 1) * ASRConstants.vocabSize, count: ASRConstants.vocabSize)

        // Pre-compute RoPE table for decode
        let maxPosition = seqLen + tokenLimit
        let (ropeCosFull, ropeSinFull) = computeRoPE(positions: Array(0 ..< maxPosition), headDim: hd)

        // Pre-allocate decode input arrays (reused each step)
        let hiddenArr = try makeZeroArray(shape: [1, 1, hs])
        let cosStepArr = try makeZeroArray(shape: [1, 1, hd])
        let sinStepArr = try makeZeroArray(shape: [1, 1, hd])


        // Autoregressive generation
        var generatedIds: [Int] = []
        var position = seqLen

        for _ in 0 ..< tokenLimit {
            if nextToken == ASRConstants.eosID || nextToken == ASRConstants.imEndID { break }
            generatedIds.append(nextToken)

            // Embed token
            let embed = embedTokens.lookup([nextToken])
            fillMultiArray(hiddenArr, with: embed)

            // RoPE for this position
            let cosOff = position * hd
            let sinOff = position * hd
            fillMultiArray(cosStepArr, with: Array(ropeCosFull[cosOff ..< cosOff + hd]))
            fillMultiArray(sinStepArr, with: Array(ropeSinFull[sinOff ..< sinOff + hd]))

            // Causal mask: all zeros for single-token decode (attend to all previous)
            let mask = try makeZeroArray(shape: [1, 1, 1, position + 1])

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": hiddenArr,
                "position_cos": cosStepArr,
                "position_sin": sinStepArr,
                "attention_mask": mask,
            ] as [String: MLMultiArray])

            let out = try await lmDecoder.prediction(from: input, using: state)
            guard let stepLogits = out.featureValue(for: "logits")?.multiArrayValue else {
                throw VibeVoiceError.modelError("Missing logits from LM step")
            }
            nextToken = argmaxMultiArray(stepLogits, lastTokenOffset: 0, count: ASRConstants.vocabSize)
            position += 1
        }

        let genTime = CFAbsoluteTimeGetCurrent() - t0

        // Decode tokens to text
        let tokenizer = try await TokenizerLoader.load(from: tokenizerDir)
        let transcription = tokenizer.decode(tokens: generatedIds)

        return TranscriptionResult(
            text: transcription,
            audioDuration: durationSecs,
            tokensGenerated: generatedIds.count,
            generationTime: genTime
        )
    }

    // MARK: - Audio Loading

    private func loadAudio(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(ASRConstants.sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw VibeVoiceError.modelError("Failed to create audio format")
        }

        let originalFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        // Read in original format
        guard let originalBuffer = AVAudioPCMBuffer(pcmFormat: originalFormat, frameCapacity: frameCount) else {
            throw VibeVoiceError.modelError("Failed to create audio buffer")
        }
        try file.read(into: originalBuffer)

        // Convert to mono 24kHz
        guard let converter = AVAudioConverter(from: originalFormat, to: format) else {
            throw VibeVoiceError.modelError("Failed to create audio converter")
        }

        let ratio = Double(ASRConstants.sampleRate) / originalFormat.sampleRate
        let outputFrames = AVAudioFrameCount(Double(frameCount) * ratio) + 1
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: outputFrames) else {
            throw VibeVoiceError.modelError("Failed to create output buffer")
        }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return originalBuffer
        }
        if let error { throw error }

        guard let channelData = outputBuffer.floatChannelData else {
            throw VibeVoiceError.modelError("No audio channel data")
        }
        let sampleCount = Int(outputBuffer.frameLength)
        var wav = Array(UnsafeBufferPointer(start: channelData[0], count: sampleCount))

        // Normalize to -25 dB FS (RMS-based)
        var sumSq: Float = 0
        for s in wav { sumSq += s * s }
        let rms = sqrt(sumSq / Float(wav.count))
        if rms > 0 {
            let targetRMS: Float = pow(10, -25.0 / 20.0)
            let scale = targetRMS / rms
            for i in wav.indices { wav[i] *= scale }
        }

        // Pad to multiple of hopLength
        let remainder = wav.count % ASRConstants.hopLength
        if remainder > 0 {
            wav.append(contentsOf: [Float](repeating: 0, count: ASRConstants.hopLength - remainder))
        }

        return wav
    }

    // MARK: - Audio Encoding

    private func encodeAudio(wav: [Float]) throws -> [Float] {
        let numChunks = (wav.count + ASRConstants.chunkSamples - 1) / ASRConstants.chunkSamples
        var allAcoustic: [Float] = []  // accumulate (T, vaeDim)
        var allSemantic: [Float] = []  // accumulate (T, semDim)
        var totalTokens = 0

        for c in 0 ..< numChunks {
            let start = c * ASRConstants.chunkSamples
            let end = min(start + ASRConstants.chunkSamples, wav.count)

            // Zero-padded chunk
            var chunk = [Float](repeating: 0, count: ASRConstants.chunkSamples)
            for i in 0 ..< (end - start) {
                chunk[i] = wav[start + i]
            }

            let audioInput = try makeMultiArray(chunk, shape: [1, 1, ASRConstants.chunkSamples])
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "audio": audioInput,
            ] as [String: MLMultiArray])

            let out = try fusedEncoder.prediction(from: input)
            guard let acFeatures = out.featureValue(for: "acoustic_features")?.multiArrayValue,
                  let semFeatures = out.featureValue(for: "semantic_features")?.multiArrayValue else {
                throw VibeVoiceError.modelError("Missing encoder features")
            }

            // Trim to actual tokens for this chunk
            let chunkTokens = min(
                ASRConstants.chunkTokens,
                (end - start + ASRConstants.hopLength - 1) / ASRConstants.hopLength
            )

            let acData = readMultiArray(acFeatures)
            let semData = readMultiArray(semFeatures)

            // Append trimmed features
            for t in 0 ..< chunkTokens {
                for d in 0 ..< ASRConstants.vaeDim {
                    allAcoustic.append(acData[t * ASRConstants.vaeDim + d])
                }
                for d in 0 ..< ASRConstants.semDim {
                    allSemantic.append(semData[t * ASRConstants.semDim + d])
                }
            }
            totalTokens += chunkTokens
        }

        // VAE noise injection (matches Python: per-batch scale * per-element noise)
        var rng = NumpyRNG(seed: 42)
        let batchScale = ASRConstants.vaeStd * rng.nextGauss()
        for i in allAcoustic.indices {
            allAcoustic[i] += batchScale * rng.nextGauss()
        }

        // Project to LM space using fused projector
        let acInput = try makeMultiArray(allAcoustic, shape: [1, totalTokens, ASRConstants.vaeDim])
        let semInput = try makeMultiArray(allSemantic, shape: [1, totalTokens, ASRConstants.semDim])

        let projInput = try MLDictionaryFeatureProvider(dictionary: [
            "acoustic_features": acInput,
            "semantic_features": semInput,
        ] as [String: MLMultiArray])

        let projOut = try fusedProjector.prediction(from: projInput)
        guard let embedding = projOut.featureValue(for: "embedding")?.multiArrayValue else {
            throw VibeVoiceError.modelError("Missing embedding from projector")
        }

        return readMultiArray(embedding)  // (totalTokens * hiddenSize)
    }

    // MARK: - Prompt Construction

    private func buildPromptIds(numAudioTokens: Int, durationSecs: Double, prompt: String?) -> [Int] {
        let tokenizer = try! SpeechToText.loadTokenizerSync(from: tokenizerDir)

        let userText: String
        if let prompt {
            userText = "This is a \(String(format: "%.2f", durationSecs)) seconds audio, with extra info: \(prompt)\nPlease transcribe it with these keys: Start time, End time, Speaker ID, Content"
        } else {
            userText = "This is a \(String(format: "%.2f", durationSecs)) seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        }

        let audioSection = [ASRConstants.audioBosID]
            + [Int](repeating: ASRConstants.audioTokenID, count: numAudioTokens)
            + [ASRConstants.audioEosID]

        let userTextIds = tokenizer.encode(text: "\n\(userText)")

        return promptPrefix
            + audioSection + userTextIds + [ASRConstants.imEndID, nlToken]
            + promptSuffix
    }
}
