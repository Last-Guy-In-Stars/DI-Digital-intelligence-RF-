#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "Собираю Leta.app…"
mkdir -p build/Leta.app/Contents/MacOS
cp tools/LetaInfo.plist build/Leta.app/Contents/Info.plist
swiftc -O -o build/Leta.app/Contents/MacOS/Leta tools/LetaApp.swift \
  -framework Cocoa -framework AVFoundation -framework Network -framework UserNotifications

echo "Подписываю…"
codesign --force --deep -s - build/Leta.app

echo "Устанавливаю в Программы…"
pkill -f "Leta.app/Contents/MacOS/Leta" 2>/dev/null || true
sleep 1
rm -rf /Applications/Leta.app
cp -R build/Leta.app /Applications/Leta.app

echo "Запускаю…"
open /Applications/Leta.app
echo "Готово. Leta.app — в Программах (и Dock), сборка — в build/."
