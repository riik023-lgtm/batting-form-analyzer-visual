import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '打撃フォーム可視化分析｜8項目',
  description: '比較選手と自分の打撃フォームを8項目すべて解析し、計測箇所を画像で可視化します。',
};
export const viewport: Viewport = { width:'device-width', initialScale:1, viewportFit:'cover', themeColor:'#10243a' };
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="ja"><body>{children}</body></html>}
