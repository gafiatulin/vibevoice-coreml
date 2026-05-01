import Foundation

// MARK: - Embedding Table (.bin format)

/// Loads float16 embedding tables with uint32 header: [dim0, dim1, ...] [fp16 data].
struct EmbeddingTable {
    let data: [Float]   // flattened float32
    let rows: Int
    let cols: Int

    init(contentsOf url: URL) throws {
        let raw = try Data(contentsOf: url)
        guard raw.count >= 8 else {
            throw VibeVoiceError.invalidData("Embedding file too small")
        }

        let dim0 = raw.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
        let dim1 = raw.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
        rows = Int(dim0)
        cols = Int(dim1)

        let headerSize = 8
        let expectedBytes = rows * cols * 2  // float16
        guard raw.count >= headerSize + expectedBytes else {
            throw VibeVoiceError.invalidData("Embedding file truncated: expected \(headerSize + expectedBytes), got \(raw.count)")
        }

        // Convert float16 → float32 using Swift's native Float16
        // Explicit type annotations on all intermediate bindings to avoid Swift WMO
        // compiler crash (whole-module optimization + type inference over pointer casts).
        let totalElems = rows * cols
        let floats: [Float] = raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
            let base: UnsafeRawPointer = ptr.baseAddress!.advanced(by: headerSize)
            let fp16Ptr: UnsafePointer<Float16> = base.assumingMemoryBound(to: Float16.self)
            let fp16Buffer: UnsafeBufferPointer<Float16> = UnsafeBufferPointer(start: fp16Ptr, count: totalElems)
            return fp16Buffer.map { Float($0) }
        }
        data = floats
    }

    /// Look up embeddings for token IDs. Returns flat array of (count * cols) floats.
    func lookup(_ tokenIds: [Int]) -> [Float] {
        var result = [Float](repeating: 0, count: tokenIds.count * cols)
        for (i, id) in tokenIds.enumerated() {
            let srcOffset = id * cols
            let dstOffset = i * cols
            for j in 0 ..< cols {
                result[dstOffset + j] = data[srcOffset + j]
            }
        }
        return result
    }
}

// MARK: - Voice Prompt (.vvvoice format)

struct VoicePromptSection {
    let kCache: [Float]     // float32, shape (totalKV, seqLen, headDim)
    let vCache: [Float]     // float32, same shape
    let lastHidden: [Float] // float32, shape (hiddenDim,)
    let meta: VoicePromptMeta
}

struct VoicePromptMeta {
    let numLayers: Int
    let numKVHeads: Int
    let seqLen: Int
    let headDim: Int
    let hiddenDim: Int
}

struct VoicePrompt {
    let sections: [VoicePromptSection]  // [lm, tts_lm, neg_lm, neg_tts_lm]

    init(contentsOf url: URL) throws {
        let raw = try Data(contentsOf: url)
        guard raw.count >= 8 else {
            throw VibeVoiceError.invalidData("Voice prompt too small")
        }

        // Header
        let magic = String(data: raw[0 ..< 4], encoding: .ascii)
        guard magic == "VVVP" else {
            throw VibeVoiceError.invalidData("Bad magic: \(magic ?? "nil")")
        }
        let version = raw.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt16.self) }
        guard version == 1 else {
            throw VibeVoiceError.invalidData("Unsupported version: \(version)")
        }
        let numSections = Int(raw.withUnsafeBytes { $0.load(fromByteOffset: 6, as: UInt16.self) })

        // Section table
        let headerSize = 8
        let entrySize = 16
        var loadedSections: [VoicePromptSection] = []

        for i in 0 ..< numSections {
            let offset = headerSize + i * entrySize
            let numLayers = Int(raw.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt16.self) })
            let numKVHeads = Int(raw.withUnsafeBytes { $0.load(fromByteOffset: offset + 2, as: UInt16.self) })
            let seqLen = Int(raw.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: UInt32.self) })
            let headDim = Int(raw.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: UInt16.self) })
            let hiddenDim = Int(raw.withUnsafeBytes { $0.load(fromByteOffset: offset + 10, as: UInt16.self) })
            let dataOffset = Int(raw.withUnsafeBytes { $0.load(fromByteOffset: offset + 12, as: UInt32.self) })

            let meta = VoicePromptMeta(
                numLayers: numLayers, numKVHeads: numKVHeads,
                seqLen: seqLen, headDim: headDim, hiddenDim: hiddenDim
            )

            let totalKV = numLayers * numKVHeads
            let kvElems = totalKV * seqLen * headDim
            let kvBytes = kvElems * 2  // float16

            // Read K cache
            let kCache = readFloat16(from: raw, offset: dataOffset, count: kvElems)
            // Read V cache
            let vCache = readFloat16(from: raw, offset: dataOffset + kvBytes, count: kvElems)
            // Read last hidden
            let lastHidden = readFloat16(from: raw, offset: dataOffset + 2 * kvBytes, count: hiddenDim)

            loadedSections.append(VoicePromptSection(
                kCache: kCache, vCache: vCache, lastHidden: lastHidden, meta: meta
            ))
        }

        sections = loadedSections
    }
}

// MARK: - Float16 Utilities

// Explicit type annotations on all bindings to avoid Swift WMO compiler crash
// (whole-module optimization + type inference over pointer casts).
private func readFloat16(from data: Data, offset: Int, count: Int) -> [Float] {
    return data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
        let base: UnsafeRawPointer = ptr.baseAddress!.advanced(by: offset)
        let fp16Ptr: UnsafePointer<Float16> = base.assumingMemoryBound(to: Float16.self)
        let buffer: UnsafeBufferPointer<Float16> = UnsafeBufferPointer(start: fp16Ptr, count: count)
        return buffer.map { Float($0) }
    }
}
