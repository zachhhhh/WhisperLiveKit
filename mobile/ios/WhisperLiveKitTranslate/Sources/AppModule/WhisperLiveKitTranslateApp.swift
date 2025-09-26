import SwiftUI

#if canImport(UIKit)
@available(iOS 16.0, *)
@main
struct WhisperLiveKitTranslateApp: App {
    init() {
        print("Bundle ID: \(Bundle.main.bundleIdentifier ?? "nil")")
    }
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
#else
@main
struct WhisperLiveKitTranslateApp {
    static func main() {
        // Placeholder entrypoint for non-iOS toolchains.
    }
}
#endif
