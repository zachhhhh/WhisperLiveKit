// swift-tools-version: 5.9
import PackageDescription
#if canImport(AppleProductTypes)
import AppleProductTypes
#endif

let products: [Product] = {
    #if canImport(AppleProductTypes)
    return [
        .iOSApplication(
            name: "WhisperLive",
            targets: ["AppModule"],
            bundleIdentifier: "com.whisperlivekit.translate",
            teamIdentifier: "ABC123DEF4",
            displayVersion: "1.0",
            bundleVersion: "1",
            appIcon: nil,
            accentColor: nil,
            supportedDeviceFamilies: [
                .phone
            ],
            supportedInterfaceOrientations: [
                .portrait
            ],
            infoPlist: .file(path: "AppModule/Resources/Info.plist")
        )
    ]
    #else
    return [
        .executable(name: "WhisperLiveKitTranslateCLI", targets: ["AppModule"])
    ]
    #endif
}()

let package = Package(
    name: "WhisperLiveKitTranslate",
    platforms: [
        .iOS(.v16)
    ],
    products: products,
    dependencies: [],
    targets: [
        .executableTarget(
            name: "AppModule",
            path: "Sources",
            exclude: [
                "AppModule/Resources/Info.plist"
            ],
            resources: [
                .process("AppModule/Resources/Assets.xcassets")
            ]
        )
    ]
)
