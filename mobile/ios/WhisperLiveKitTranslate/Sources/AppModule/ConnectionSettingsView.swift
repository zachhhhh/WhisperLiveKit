#if canImport(UIKit)
import SwiftUI

@available(iOS 16.0, *)
struct ConnectionSettingsView: View {
    @Binding var host: String
    @Binding var port: String

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            Section("Server") {
                TextField("Host", text: $host)
                    .keyboardType(.URL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()

                TextField("Port", text: $port)
                    .keyboardType(.numberPad)
            }

            Section(footer: footerText) {
                EmptyView()
            }
        }
        .navigationTitle("Connection")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Close") { dismiss() }
            }
        }
    }

    private var footerText: some View {
        Text("The iOS app streams microphone audio over WebSocket to the WhisperLiveKit backend. Update the host and port to match your deployment.")
            .font(.footnote)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.leading)
    }
}

#if DEBUG
struct ConnectionSettingsView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            ConnectionSettingsView(host: .constant("127.0.0.1"), port: .constant("8000"))
        }
    }
}
#endif
#endif
