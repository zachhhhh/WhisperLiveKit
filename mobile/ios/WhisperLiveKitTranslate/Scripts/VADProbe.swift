#if canImport(UIKit)
import Foundation

// Reuse static helpers from AudioCapture via objc runtime once compiled

func generateSine(amplitude: Float, samples: Int, frequency: Float = 440, sampleRate: Float = 16_000) -> [Float] {
    var data: [Float] = []
    data.reserveCapacity(samples)
    let twoPiFOverFs = 2 * Float.pi * frequency / sampleRate
    for n in 0..<samples {
        data.append(amplitude * sin(twoPiFOverFs * Float(n)))
    }
    return data
}

var energyHistory: [Float] = []

func probe(amplitude: Float, label: String) {
    let samples = generateSine(amplitude: amplitude, samples: 1600)
    let stats = AudioCapture.calculateEnergy(of: samples)
    let baseline = energyHistory.min()
    let relative = AudioCapture.calculateRelativeEnergy(of: samples, relativeTo: baseline)
    energyHistory.append(stats.avg)
    if energyHistory.count > 40 { energyHistory.removeFirst(energyHistory.count - 40) }
    let threshold: Float = 0.45
    let detected = AudioCapture.isVoiceDetected(in: Array(repeating: relative, count: 20), nextBufferInSeconds: 0.1, silenceThreshold: threshold)
    print(String(format: "%@ | amp=%.3f avg=%.5f baseline=%.5f rel=%.3f threshold=%.2f detected=%@", label, amplitude, stats.avg, baseline ?? -1, relative, threshold, detected ? "true" : "false"))
}

@main
struct Runner {
    static func main() {
        for _ in 0..<5 { probe(amplitude: 0.001, label: "silence baseline") }
        probe(amplitude: 0.02, label: "breathing")
        probe(amplitude: 0.05, label: "quiet speech")
        probe(amplitude: 0.1, label: "normal speech")
        probe(amplitude: 0.3, label: "loud speech")
    }
}
#else
@main
struct Runner {
    static func main() {
        print("VADProbe requires UIKit support")
    }
}
#endif
