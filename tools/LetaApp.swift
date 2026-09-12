import Cocoa
import AVFoundation
import Network
import UserNotifications

func findJarvis() -> String {
    let bundleDir = NSString(string: Bundle.main.bundlePath).deletingLastPathComponent
    let home = NSString(string: "~").expandingTildeInPath
    let candidates = [
        bundleDir,
        home + "/Desktop/My/Jarvis",
        home + "/Jarvis",
    ]
    for c in candidates {
        let py = c + "/.venv/bin/python"
        let src = c + "/src/main.py"
        if FileManager.default.fileExists(atPath: py)
            && FileManager.default.fileExists(atPath: src)
        {
            return c
        }
    }
    return home + "/Desktop/My/Jarvis"
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSTextFieldDelegate, UNUserNotificationCenterDelegate {
    var window: NSWindow!
    var textView: NSTextView!
    var inputField: NSTextField!
    var micButton: NSButton!
    var muteButton: NSButton!
    var wakeButton: NSButton!
    var sendButton: NSButton!
    var conn: NWConnection?
    var recorder: AVAudioRecorder?
    var recording = false
    var buffer = Data()
    var busy = false
    var wakeTimer: Timer?
    var wasConnected = false

    let jarvis = findJarvis()
    var sockPath: String { jarvis + "/brain/soul.sock" }

    func applicationDidFinishLaunching(_ n: Notification) {
        setupMenu()
        setupWindow()
        setupNotifications()
        connectSoul()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool {
        false
    }

    func applicationShouldHandleReopen(_ s: NSApplication, hasVisibleWindows: Bool) -> Bool {
        if window == nil {
            setupWindow()
        }
        window.makeKeyAndOrderFront(nil)
        textView.scrollRangeToVisible(
            NSRange(location: textView.string.count, length: 0)
        )
        return true
    }

    func setupMenu() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu(title: "Leta")
        appMenu.addItem(
            NSMenuItem(
                title: "Завершить Leta",
                action: #selector(NSApplication.terminate(_:)),
                keyEquivalent: "q"
            )
        )
        appMenuItem.submenu = appMenu

        let editMenuItem = NSMenuItem()
        mainMenu.addItem(editMenuItem)
        let editMenu = NSMenu(title: "Правка")
        editMenu.addItem(
            NSMenuItem(
                title: "Копировать", action: #selector(NSText.copy(_:)),
                keyEquivalent: "c"
            )
        )
        editMenu.addItem(
            NSMenuItem(
                title: "Выделить всё", action: #selector(NSText.selectAll(_:)),
                keyEquivalent: "a"
            )
        )
        editMenu.addItem(
            NSMenuItem(
                title: "Вставить", action: #selector(NSText.paste(_:)),
                keyEquivalent: "v"
            )
        )
        editMenuItem.submenu = editMenu

        NSApp.mainMenu = mainMenu
    }

    func setupWindow() {
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 460, height: 640),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false
        )
        window.title = "Leta"
        window.center()
        window.isReleasedWhenClosed = false

        let view = NSView(frame: NSRect(x: 0, y: 0, width: 460, height: 640))

        let scroll = NSScrollView(frame: NSRect(x: 12, y: 64, width: 436, height: 520))
        scroll.autoresizingMask = [.width, .height]
        scroll.hasVerticalScroller = true
        textView = NSTextView(frame: scroll.bounds)
        textView.isEditable = false
        textView.isRichText = false
        textView.font = NSFont.systemFont(ofSize: 14)
        textView.drawsBackground = false
        textView.textContainerInset = NSSize(width: 4, height: 6)
        scroll.documentView = textView
        view.addSubview(scroll)

        inputField = NSTextField(frame: NSRect(x: 12, y: 20, width: 268, height: 30))
        inputField.placeholderString = "Напиши ей…"
        inputField.font = NSFont.systemFont(ofSize: 14)
        inputField.delegate = self
        inputField.action = #selector(sendAction)
        inputField.autoresizingMask = [.width, .maxYMargin]
        view.addSubview(inputField)

        micButton = NSButton(title: "🎙", target: self, action: #selector(micAction))
        micButton.frame = NSRect(x: 286, y: 20, width: 36, height: 30)
        micButton.bezelStyle = .rounded
        view.addSubview(micButton)

        muteButton = NSButton(title: "🔊", target: self, action: #selector(muteAction))
        muteButton.frame = NSRect(x: 328, y: 20, width: 36, height: 30)
        muteButton.bezelStyle = .rounded
        view.addSubview(muteButton)

        sendButton = NSButton(title: "➤", target: self, action: #selector(sendAction))
        sendButton.frame = NSRect(x: 370, y: 20, width: 36, height: 30)
        sendButton.bezelStyle = .rounded
        view.addSubview(sendButton)

        wakeButton = NSButton(
            title: "🌞 Разбудить её", target: self, action: #selector(wakeAction)
        )
        wakeButton.bezelStyle = .rounded
        wakeButton.font = NSFont.boldSystemFont(ofSize: 14)
        wakeButton.isHidden = true
        view.addSubview(wakeButton)

        wakeButton.translatesAutoresizingMaskIntoConstraints = false
        wakeButton.leadingAnchor.constraint(
            equalTo: view.leadingAnchor, constant: 12
        ).isActive = true
        wakeButton.trailingAnchor.constraint(
            equalTo: view.trailingAnchor, constant: -12
        ).isActive = true
        wakeButton.bottomAnchor.constraint(
            equalTo: view.bottomAnchor, constant: -20
        ).isActive = true
        wakeButton.heightAnchor.constraint(equalToConstant: 34).isActive = true

        window.contentView = view
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func setupNotifications() {
        UNUserNotificationCenter.current().delegate = self
        let nc = UNUserNotificationCenter.current()
        nc.requestAuthorization(options: [.alert, .sound]) { granted, error in
            nc.getNotificationSettings { s in
                let names = [0: "не запрошено", 1: "ЗАПРЕЩЕНО", 2: "разрешено", 3: "временно"]
                let st = names[s.authorizationStatus.rawValue] ?? "?"
                let line = "auth=\(s.authorizationStatus.rawValue) granted=\(granted) err=\(error?.localizedDescription ?? "-")"
                try? line.write(toFile: "/tmp/leta_push.txt", atomically: true, encoding: .utf8)
                DispatchQueue.main.async {
                    if s.authorizationStatus.rawValue == 2 {
                        self.append("system", "Уведомления разрешены ✓")
                    } else {
                        self.append(
                            "system",
                            "Уведомления: \(st). Открой Системные настройки → Уведомления → Leta и разреши."
                        )
                    }
                }
            }
        }
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        DispatchQueue.main.async {
            NSApp.activate(ignoringOtherApps: true)
            self.window.makeKeyAndOrderFront(nil)
            self.textView.scrollRangeToVisible(
                NSRange(location: self.textView.string.count, length: 0)
            )
        }
        completionHandler()
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
    }

    func connectSoul() {
        if !wasConnected {
            append("system", "Ищу её…")
        }
        let endpoint = NWEndpoint.unix(path: sockPath)
        let c = NWConnection(to: endpoint, using: NWParameters.tcp)
        conn = c
        c.stateUpdateHandler = { [weak self] state in
            DispatchQueue.main.async {
                switch state {
                case .ready:
                    self?.soulConnected()
                case .failed, .cancelled:
                    self?.soulAsleep()
                default:
                    break
                }
            }
        }
        c.start(queue: .global())
    }

    func soulConnected() {
        wasConnected = true
        wakeTimer?.invalidate()
        wakeButton.isHidden = true
        inputField.isHidden = false
        micButton.isHidden = false
        muteButton.isHidden = false
        sendButton.isHidden = false
        inputField.isEnabled = true
        window.title = "Leta"
        append("system", "Она жива — вы на связи.")
        sendLine(["type": "hello", "client": "app"])
        receiveLoop()
    }

    func soulAsleep() {
        conn = nil
        wakeButton.isHidden = false
        inputField.isHidden = true
        micButton.isHidden = true
        muteButton.isHidden = true
        sendButton.isHidden = true
        inputField.isEnabled = false
        if wasConnected {
            window.title = "Leta — ищу её…"
            append("system", "Связь с ней прервалась — ищу её снова…")
            startReconnect()
        } else {
            window.title = "Leta — спит"
            append("system", "Она спит — её жизнь не запущена.")
        }
    }

    func startReconnect() {
        wakeTimer?.invalidate()
        var tries = 0
        wakeTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] t in
            guard let self = self else { t.invalidate(); return }
            if FileManager.default.fileExists(atPath: self.sockPath) {
                t.invalidate()
                self.connectSoul()
            } else {
                tries += 1
                if tries > 120 {
                    t.invalidate()
                    self.wasConnected = false
                    self.window.title = "Leta — спит"
                    self.append("system", "Её долго нет. Разбуди кнопкой, когда захочешь.")
                }
            }
        }
    }

    @objc func wakeAction() {
        append("system", "Будужу её — она просыпается около минуты…")
        wakeButton.isHidden = true
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-c", "cd '" + jarvis + "' && ./leta start"]
        try? p.run()
        var tries = 0
        wakeTimer?.invalidate()
        wakeTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] t in
            tries += 1
            guard let self = self else { t.invalidate(); return }
            if FileManager.default.fileExists(atPath: self.sockPath) {
                t.invalidate()
                self.connectSoul()
            } else if tries > 40 {
                t.invalidate()
                self.append("system", "Не смогла проснуться. Запусти её: ./leta start")
                self.wakeButton.isHidden = false
            }
        }
    }

    func receiveLoop() {
        conn?.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            if let d = data, !d.isEmpty {
                self?.consume(d)
            }
            if isComplete || error != nil {
                DispatchQueue.main.async { self?.soulAsleep() }
                return
            }
            self?.receiveLoop()
        }
    }

    func consume(_ data: Data) {
        buffer.append(data)
        while let nl = buffer.firstIndex(of: UInt8(ascii: "\n")) {
            let lineData = buffer[buffer.startIndex..<nl]
            buffer = Data(buffer[buffer.index(after: nl)...])
            guard let line = String(data: Data(lineData), encoding: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
            else { continue }
            DispatchQueue.main.async { self.handle(obj) }
        }
    }

    func handle(_ obj: [String: Any]) {
        let type = obj["type"] as? String ?? ""
        switch type {
        case "answer":
            let text = obj["text"] as? String ?? ""
            append("leta", text)
            if !NSApp.isActive || !window.isKeyWindow {
                notifyUser(text)
            }
            busy = false
            window.title = "Leta"
            inputField.isEnabled = true
        case "heard":
            append("you", obj["text"] as? String ?? "")
        case "thinking":
            busy = true
            window.title = "Leta — думает…"
        case "muted":
            let v = obj["value"] as? Bool ?? false
            muteButton.title = v ? "🔇" : "🔊"
        default:
            break
        }
    }

    func append(_ who: String, _ text: String) {
        let prefix: String
        let color: NSColor
        switch who {
        case "you": prefix = "Ты: "; color = .systemYellow
        case "leta": prefix = "Leta: "; color = .systemTeal
        default: prefix = ""; color = .systemGray
        }
        let para = NSMutableParagraphStyle()
        para.paragraphSpacing = 8
        let s = NSAttributedString(string: prefix + text + "\n", attributes: [
            .foregroundColor: color,
            .font: NSFont.systemFont(ofSize: 14),
            .paragraphStyle: para,
        ])
        textView.textStorage?.append(s)
        textView.scrollRangeToVisible(NSRange(location: textView.string.count, length: 0))
    }

    func notifyUser(_ text: String) {
        let content = UNMutableNotificationContent()
        content.title = "Leta написала"
        content.body = String(text.prefix(180))
        content.sound = .default
        let req = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(req)
    }

    func send(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, conn != nil, !busy else { return }
        append("you", t)
        inputField.stringValue = ""
        sendLine(["type": "user", "text": t])
    }

    func sendLine(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let nl = "\n".data(using: .utf8)
        else { return }
        conn?.send(content: data + nl, completion: .contentProcessed { _ in })
    }

    @objc func sendAction() {
        send(inputField.stringValue)
    }

    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            send(inputField.stringValue)
            return true
        }
        return false
    }

    @objc func muteAction() {
        sendLine(["type": "mute"])
    }

    @objc func micAction() {
        if recording {
            recorder?.stop()
            recording = false
            micButton.title = "🎙"
            let path = NSTemporaryDirectory() + "leta_voice.wav"
            if FileManager.default.fileExists(atPath: path) {
                append("system", "…слушает запись…")
                sendLine(["type": "voice", "path": path])
            }
        } else {
            let path = NSTemporaryDirectory() + "leta_voice.wav"
            try? FileManager.default.removeItem(atPath: path)
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16000.0,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
            ]
            recorder = try? AVAudioRecorder(url: URL(fileURLWithPath: path), settings: settings)
            recorder?.record()
            recording = true
            micButton.title = "⏹"
            append("system", "Говори… (⏹ — закончить)")
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
withExtendedLifetime(delegate) {
    app.run()
}
