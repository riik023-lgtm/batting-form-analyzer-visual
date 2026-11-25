# Batting Form Analyzer Visual — 8項目可視化版

既存の `batting-form-analyzer-vercel` を残したまま、別アプリとして使うためのアレンジ版です。

## 何が違うか

- 8項目を毎回すべて解析
  1. 身体の開き
  2. 割れ
  3. ボールの引きつけ
  4. 軸のブレ
  5. 前脚の壁
  6. 頭の残り
  7. 体重移動
  8. 腰の回転
- 数字だけではなく、比較選手と自分の実動画フレームに計測線を描いて表示
- 実フレームを作れなかった項目は、顔なしの簡易人体イラストへ自動fallback
- 8項目の数値、比較判定、改善アドバイスを1画面で確認
- 日本語の練習動画を優先表示し、海外動画は補助リンクとして残す
- 解析結果を `.bfa` ではなく、その端末のブラウザ内 `IndexedDB` に保存
- 複数選手を選手名で分けて履歴管理
- 「PDFで保存」ボタンはブラウザの印刷機能を開き、PDFとして保存できる
- 解析に使用した動画はVercel Private Blobへ一時アップロードするが、解析成功後に自動削除
- Blob upload routeにも1日TTLがあり、削除に失敗した場合も長期間残さない

## 保存方式

### 端末内履歴
解析結果、8項目の計測画像、選手名、日時をIndexedDBに保存します。動画ファイルは端末履歴へ保存しません。

注意: ブラウザの「サイトデータを削除」、シークレットモード、端末のストレージ整理などで消える場合があります。

### PDF
結果画面の「PDFで保存」を押すと印刷画面が開きます。PCでは「PDFとして保存」を選択してください。日本語・画像を含むため、クライアント側でPDFライブラリを追加するよりブラウザ印刷を採用しています。

### Private Blob
現構成はPython / MediaPipeで解析するため、動画そのものを完全ローカルだけで処理していません。動画はPrivate Blobへ一時アップロードし、解析成功直後に削除します。

完全ローカル化する場合は、将来 `MediaPipe Tasks Vision for Web` へ解析処理自体を移植する別改修が必要です。

## Vercel
- frontend: Next.js
- backend: FastAPI + MediaPipe + OpenCV
- backend runtime: Vercel Container Runtime
- temporary video storage: Vercel Private Blob
- persistent result storage: user browser IndexedDB

## Health check

`/api/health`

正常時の例:

```json
{
  "ok": true,
  "version": "visual-1.0",
  "runtime": "python",
  "features": 8,
  "measurement_visuals": true,
  "local_result_storage": "browser-indexeddb"
}
```

## 既存アプリとの関係

これは既存の打撃フォーム分析を上書きするためのものではありません。
新しいGitHubリポジトリ、例 `batting-form-analyzer-visual` を作り、別のVercel Projectとしてデプロイしてください。
