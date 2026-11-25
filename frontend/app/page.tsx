'use client';

import { upload } from '@vercel/blob/client';
import { useEffect, useMemo, useRef, useState } from 'react';

const FEATURE_OPTIONS = [
  ['身体を開くな', '身体の開き'],
  ['割れを作れ', '割れ'],
  ['ボールを引きつけろ', 'ボールの引きつけ'],
  ['軸をぶらすな', '軸のブレ'],
  ['壁を作れ', '前脚の壁'],
  ['頭を残せ', '頭の残り'],
  ['体重移動をしろ', '体重移動'],
  ['腰を使え', '腰の回転'],
] as const;
const ALL_FEATURE_KEYS = FEATURE_OPTIONS.map(([key]) => key);

const FEATURE_HINTS: Record<string, string> = {
  '身体を開くな': '肩の回転量',
  '割れを作れ': '肩と骨盤のひねり差',
  'ボールを引きつけろ': '手と体幹の距離',
  '軸をぶらすな': '体幹中心の横移動',
  '壁を作れ': '前脚の膝角度',
  '頭を残せ': '頭部の移動量',
  '体重移動をしろ': '骨盤中心の水平移動',
  '腰を使え': '骨盤の回転速度',
};

type Roi = { x: number; y: number; width: number; height: number };
type DragMode = 'move' | 'nw' | 'ne' | 'sw' | 'se';
type DragState = { mode: DragMode; startX: number; startY: number; startRoi: Roi };
type AnalysisResult = Record<string, any>;
type SavedResult = {
  id: string;
  playerName: string;
  batterSide: '右打者' | '左打者';
  createdAt: string;
  result: AnalysisResult;
};

const DEFAULT_ROI: Roi = { x: 0.08, y: 0.08, width: 0.84, height: 0.84 };
const PORTRAIT_ROI: Roi = { x: 0.05, y: 0.12, width: 0.90, height: 0.82 };
const LANDSCAPE_ROI: Roi = { x: 0.08, y: 0.08, width: 0.84, height: 0.84 };
const MIN_ROI_WIDTH = 0.16;
const MIN_ROI_HEIGHT = 0.20;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
function normalizeRoi(input: Partial<Roi>): Roi {
  const width = clamp(Number(input.width ?? DEFAULT_ROI.width), MIN_ROI_WIDTH, 1);
  const height = clamp(Number(input.height ?? DEFAULT_ROI.height), MIN_ROI_HEIGHT, 1);
  let x = clamp(Number(input.x ?? DEFAULT_ROI.x), 0, 1 - width);
  let y = clamp(Number(input.y ?? DEFAULT_ROI.y), 0, 1 - height);
  if (x + width > 1) x = 1 - width;
  if (y + height > 1) y = 1 - height;
  return { x, y, width, height };
}
function resizeRoi(start: Roi, mode: DragMode, dx: number, dy: number): Roi {
  const r = normalizeRoi(start);
  const right = r.x + r.width;
  const bottom = r.y + r.height;
  if (mode === 'move') {
    return { ...r, x: clamp(r.x + dx, 0, 1 - r.width), y: clamp(r.y + dy, 0, 1 - r.height) };
  }
  let x0 = r.x, y0 = r.y, x1 = right, y1 = bottom;
  if (mode.includes('w')) x0 = clamp(r.x + dx, 0, x1 - MIN_ROI_WIDTH);
  if (mode.includes('e')) x1 = clamp(right + dx, x0 + MIN_ROI_WIDTH, 1);
  if (mode.includes('n')) y0 = clamp(r.y + dy, 0, y1 - MIN_ROI_HEIGHT);
  if (mode.includes('s')) y1 = clamp(bottom + dy, y0 + MIN_ROI_HEIGHT, 1);
  return normalizeRoi({ x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
}
function uid() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
function fileUrl(file: File | null) { return file ? URL.createObjectURL(file) : ''; }

async function readJsonResponse(response: Response) {
  const text = await response.text();
  let data: any;
  try { data = text ? JSON.parse(text) : {}; }
  catch { throw new Error(response.ok ? '解析サーバーから読み取れない応答が返されました。' : `解析サーバーでエラーが発生しました（HTTP ${response.status}）。`); }
  if (!response.ok) throw new Error(data?.detail || data?.error || `解析に失敗しました（HTTP ${response.status}）。`);
  return data;
}

const DB_NAME = 'batting-form-analyzer-visual';
const DB_STORE = 'results';
function openResultDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(DB_STORE)) {
        const store = db.createObjectStore(DB_STORE, { keyPath: 'id' });
        store.createIndex('createdAt', 'createdAt');
        store.createIndex('playerName', 'playerName');
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function saveResultLocal(record: SavedResult) {
  const db = await openResultDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.objectStore(DB_STORE).put(record);
    tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error);
  });
  db.close();
}
async function listResultsLocal(): Promise<SavedResult[]> {
  const db = await openResultDb();
  const rows = await new Promise<SavedResult[]>((resolve, reject) => {
    const req = db.transaction(DB_STORE, 'readonly').objectStore(DB_STORE).getAll();
    req.onsuccess = () => resolve((req.result || []) as SavedResult[]);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return rows.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}
async function deleteResultLocal(id: string) {
  const db = await openResultDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readwrite'); tx.objectStore(DB_STORE).delete(id);
    tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error);
  });
  db.close();
}

function VideoCropCard({
  title, file, roi, onChange, impactTime, impactPreview, onImpactSet, onImpactClear,
}: {
  title: string; file: File | null; roi: Roi; onChange: (roi: Roi) => void;
  impactTime: number | null; impactPreview: string;
  onImpactSet: (time: number, preview: string) => void; onImpactClear: () => void;
}) {
  const url = useMemo(() => fileUrl(file), [file]);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const initializedUrlRef = useRef('');
  const [drag, setDrag] = useState<DragState | null>(null);
  const [videoSize, setVideoSize] = useState({ width: 16, height: 9 });
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  useEffect(() => {
    if (!drag) return;
    const move = (event: PointerEvent) => {
      const stage = stageRef.current; if (!stage) return;
      event.preventDefault(); const rect = stage.getBoundingClientRect(); if (!rect.width || !rect.height) return;
      onChange(resizeRoi(drag.startRoi, drag.mode, (event.clientX-drag.startX)/rect.width, (event.clientY-drag.startY)/rect.height));
    };
    const up = () => setDrag(null);
    window.addEventListener('pointermove', move, { passive: false });
    window.addEventListener('pointerup', up); window.addEventListener('pointercancel', up);
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up); };
  }, [drag, onChange]);

  function beginDrag(event: React.PointerEvent, mode: DragMode) {
    event.preventDefault(); event.stopPropagation();
    setDrag({ mode, startX: event.clientX, startY: event.clientY, startRoi: normalizeRoi(roi) });
  }
  function handleMetadata(event: React.SyntheticEvent<HTMLVideoElement>) {
    const video = event.currentTarget;
    const width = Math.max(1, video.videoWidth || 16), height = Math.max(1, video.videoHeight || 9);
    setVideoSize({ width, height }); setDuration(Number.isFinite(video.duration) ? video.duration : 0); setCurrentTime(video.currentTime || 0);
    if (url && initializedUrlRef.current !== url) { initializedUrlRef.current = url; onChange(height > width ? PORTRAIT_ROI : LANDSCAPE_ROI); }
  }
  function seekBy(delta: number) {
    const video = videoRef.current; if (!video) return; video.pause();
    const max = Number.isFinite(video.duration) ? Math.max(0, video.duration - 0.001) : Math.max(0, currentTime + delta);
    const next = clamp(video.currentTime + delta, 0, max); video.currentTime = next; setCurrentTime(next);
  }
  function captureImpact() {
    const video = videoRef.current; if (!video || !video.videoWidth || !video.videoHeight) return; video.pause();
    const safe = normalizeRoi(roi); const sx = Math.round(safe.x * video.videoWidth), sy = Math.round(safe.y * video.videoHeight);
    const sw = Math.max(2, Math.round(safe.width * video.videoWidth)), sh = Math.max(2, Math.round(safe.height * video.videoHeight));
    const canvas = document.createElement('canvas'); const scale = Math.min(1, 600/sw);
    canvas.width = Math.max(1, Math.round(sw*scale)); canvas.height = Math.max(1, Math.round(sh*scale));
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    onImpactSet(video.currentTime, canvas.toDataURL('image/jpeg', .86));
  }
  const safeRoi = normalizeRoi(roi); const portrait = videoSize.height > videoSize.width;
  return <section className="card crop-card">
    <div className="section-title-row"><h3>{title}</h3><span className="muted">赤枠内だけを解析</span></div>
    <p className="crop-help">四隅の丸を動かし、スイング中の全身が入るように囲んでください。中央の「＋」で枠全体を動かせます。</p>
    {file ? <>
      <div className={`crop-stage-shell ${portrait?'portrait':'landscape'}`}><div className="crop-stage" ref={stageRef} style={{aspectRatio:`${videoSize.width} / ${videoSize.height}`}}>
        <video ref={videoRef} className="crop-video" src={url} controls playsInline preload="metadata" onLoadedMetadata={handleMetadata} onTimeUpdate={e=>setCurrentTime(e.currentTarget.currentTime)} onSeeked={e=>setCurrentTime(e.currentTarget.currentTime)}/>
        <div className="crop-box" style={{left:`${safeRoi.x*100}%`,top:`${safeRoi.y*100}%`,width:`${safeRoi.width*100}%`,height:`${safeRoi.height*100}%`}}>
          <button type="button" aria-label="左上" className="crop-handle nw" onPointerDown={e=>beginDrag(e,'nw')}/><button type="button" aria-label="右上" className="crop-handle ne" onPointerDown={e=>beginDrag(e,'ne')}/>
          <button type="button" aria-label="左下" className="crop-handle sw" onPointerDown={e=>beginDrag(e,'sw')}/><button type="button" aria-label="右下" className="crop-handle se" onPointerDown={e=>beginDrag(e,'se')}/>
          <button type="button" aria-label="枠を移動" className="crop-move" onPointerDown={e=>beginDrag(e,'move')}>＋</button>
        </div>
      </div></div>
      <div className="impact-setter"><div className="impact-setter-head"><div><strong>インパクトを指定</strong><p>動画を止めて、バットがボールに当たる瞬間に合わせます。</p></div><span>{currentTime.toFixed(3)}秒{duration>0?` / ${duration.toFixed(2)}秒`:''}</span></div>
        <div className="impact-nudge-row"><button type="button" onClick={()=>seekBy(-.033)}>−0.03秒</button><button type="button" className="impact-set-button" onClick={captureImpact}>この瞬間をインパクトに設定</button><button type="button" onClick={()=>seekBy(.033)}>＋0.03秒</button></div>
        {impactTime!==null&&impactPreview?<div className="impact-confirmed"><div className="impact-confirmed-title"><div><b>設定したインパクト</b><span>{impactTime.toFixed(3)}秒</span></div><button type="button" onClick={onImpactClear}>解除</button></div><img src={impactPreview} alt={`${title}のインパクト`}/></div>:<div className="impact-auto-note">未指定の場合は自動推定を使います。</div>}
      </div>
    </>:<div className="empty-video">動画を選択してください</div>}
  </section>;
}

function MetricDiagram({ featureKey }: { featureKey: string }) {
  const common = <><line x1="70" y1="45" x2="130" y2="45"/><line x1="82" y1="82" x2="118" y2="82"/><line x1="70" y1="45" x2="82" y2="82"/><line x1="130" y1="45" x2="118" y2="82"/><line x1="82" y1="82" x2="72" y2="135"/><line x1="118" y1="82" x2="128" y2="135"/></>;
  return <svg className="metric-diagram" viewBox="0 0 200 160" role="img" aria-label={`${featureKey}の計測イラスト`}>
    <g className="body-model"><circle cx="100" cy="23" r="12"/>{common}</g>
    <g className="measure-model">
      {featureKey==='身体を開くな'&&<><line x1="68" y1="44" x2="132" y2="44"/><path d="M72 60 Q100 25 130 58"/></>}
      {featureKey==='割れを作れ'&&<><line x1="66" y1="42" x2="134" y2="50"/><line x1="78" y1="85" x2="122" y2="78"/><path d="M89 61 Q103 49 116 61"/></>}
      {featureKey==='ボールを引きつけろ'&&<><circle cx="145" cy="68" r="5"/><circle cx="100" cy="65" r="5"/><line x1="105" y1="65" x2="140" y2="68"/></>}
      {featureKey==='軸をぶらすな'&&<><line x1="100" y1="35" x2="100" y2="118"/><path d="M86 105 L116 105"/><path d="M110 98 L118 105 L110 112"/></>}
      {featureKey==='壁を作れ'&&<><line x1="118" y1="82" x2="128" y2="118"/><line x1="128" y1="118" x2="142" y2="150"/><path d="M118 119 Q128 108 139 119"/></>}
      {featureKey==='頭を残せ'&&<><circle cx="100" cy="23" r="17"/><line x1="70" y1="23" x2="130" y2="23"/><path d="M122 16 L132 23 L122 30"/></>}
      {featureKey==='体重移動をしろ'&&<><circle cx="100" cy="82" r="6"/><line x1="80" y1="112" x2="145" y2="112"/><path d="M137 104 L147 112 L137 120"/></>}
      {featureKey==='腰を使え'&&<><line x1="78" y1="82" x2="122" y2="82"/><path d="M70 95 Q100 118 132 94"/><path d="M124 88 L134 94 L126 102"/></>}
    </g>
  </svg>;
}

function MeasurementCard({ row, visual }: { row: any; visual: any }) {
  const key = row?.['評価項目'] || visual?.key || '';
  const display = row?.['表示名'] || visual?.display_name || key;
  return <article className="measurement-card">
    <div className="measurement-head"><div><span className="metric-number">{FEATURE_OPTIONS.findIndex(v=>v[0]===key)+1}</span><h3>{display}</h3></div><span className={`level ${row?.['比較判定']||''}`}>{row?.['比較判定']||'評価'}</span></div>
    <p className="measure-how">{visual?.how_measured || FEATURE_HINTS[key]}</p>
    <div className="measure-images">
      <div><b>比較選手</b>{visual?.reference_image?<img src={visual.reference_image} alt={`${display} 比較選手の計測画像`}/>:<MetricDiagram featureKey={key}/>}</div>
      <div><b>自分</b>{visual?.target_image?<img src={visual.target_image} alt={`${display} 自分の計測画像`}/>:<MetricDiagram featureKey={key}/>}</div>
    </div>
    <div className="measure-values"><div><span>比較選手</span><strong>{row?.['比較選手表示'] ?? '-'}</strong></div><span className="measure-arrow">→</span><div><span>自分</span><strong>{row?.['自分表示'] ?? '-'}</strong></div></div>
    <p className="measure-direction">{row?.['差の方向']}</p>
    <small>{visual?.image_note || row?.['計測内容']}</small>
  </article>;
}

function QualityBadge({ label }: { label?: string }) {
  const safe=label||'未評価'; return <span className={safe==='高'?'quality good':'quality warn'}>骨格検出品質：{safe}</span>;
}

export default function Home() {
  const [playerName,setPlayerName]=useState('');
  const [batterSide,setBatterSide]=useState<'右打者'|'左打者'>('右打者');
  const [referenceFile,setReferenceFile]=useState<File|null>(null), [targetFile,setTargetFile]=useState<File|null>(null);
  const [referenceRoi,setReferenceRoi]=useState<Roi>(DEFAULT_ROI), [targetRoi,setTargetRoi]=useState<Roi>(DEFAULT_ROI);
  const [referenceBlobUrl,setReferenceBlobUrl]=useState(''), [targetBlobUrl,setTargetBlobUrl]=useState('');
  const [referenceImpactTime,setReferenceImpactTime]=useState<number|null>(null), [targetImpactTime,setTargetImpactTime]=useState<number|null>(null);
  const [referenceImpactPreview,setReferenceImpactPreview]=useState(''), [targetImpactPreview,setTargetImpactPreview]=useState('');
  const [uploadProgress,setUploadProgress]=useState(0), [status,setStatus]=useState(''), [error,setError]=useState('');
  const [result,setResult]=useState<AnalysisResult|null>(null), [history,setHistory]=useState<SavedResult[]>([]), [historyError,setHistoryError]=useState('');

  async function refreshHistory(){ try{setHistory(await listResultsLocal());setHistoryError('')}catch{setHistoryError('このブラウザでは端末内履歴を読み込めませんでした。')} }
  useEffect(()=>{refreshHistory()},[]);
  const knownPlayers=Array.from(new Set(history.map(h=>h.playerName).filter(Boolean)));

  async function uploadVideo(file:File,role:string,start:number,span:number){
    const blob=await upload(`batting-visual/${role}-${Date.now()}-${file.name}`,file,{access:'private',handleUploadUrl:'/blob-upload',multipart:true,onUploadProgress:({percentage})=>setUploadProgress(start+percentage*span/100)});return blob.url;
  }
  async function ensureBlobUrls(){ if(!referenceFile||!targetFile)throw new Error('比較選手と自分の動画を2本とも選択してください。');let r=referenceBlobUrl,t=targetBlobUrl;if(!r)r=await uploadVideo(referenceFile,'reference',0,50);if(!t)t=await uploadVideo(targetFile,'target',50,50);setReferenceBlobUrl(r);setTargetBlobUrl(t);return{r,t}; }
  async function deleteTempBlobs(urls:string[],silent=false){
    const clean=urls.filter(Boolean); if(!clean.length)return;
    try{await fetch('/blob-delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({urls:clean})});if(!silent)setStatus('一時アップロード動画を削除しました。')}catch{}
    setReferenceBlobUrl('');setTargetBlobUrl('');
  }
  async function suggestRoi(){setError('');setStatus('動画をアップロードしています…');setUploadProgress(0);try{const{r,t}=await ensureBlobUrls();setStatus('打者位置を自動推定しています…');const[a,b]=await Promise.all([
    fetch('/api/suggest-roi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_url:r})}),
    fetch('/api/suggest-roi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_url:t})})]);const[x,y]=await Promise.all([readJsonResponse(a),readJsonResponse(b)]);setReferenceRoi(normalizeRoi(x));setTargetRoi(normalizeRoi(y));setStatus('赤枠を確認し、必要なら四隅を動かしてください。')}catch(e){setError(e instanceof Error?e.message:'自動推定に失敗しました。');setStatus('')}}
  async function storeCurrentResult(data:AnalysisResult){
    const name=playerName.trim(); if(!name)return;
    const record:SavedResult={id:uid(),playerName:name,batterSide,createdAt:new Date().toISOString(),result:data};
    await saveResultLocal(record); await refreshHistory();
  }
  async function analyze(){setError('');setResult(null);setUploadProgress(0);let urls:string[]=[];try{setStatus('動画をアップロードしています…');const{r,t}=await ensureBlobUrls();urls=[r,t];setStatus('8項目を高精度解析しています…');const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reference_url:r,target_url:t,batter_side:batterSide,selected_features:ALL_FEATURE_KEYS,reference_roi:normalizeRoi(referenceRoi),target_roi:normalizeRoi(targetRoi),reference_impact_time:referenceImpactTime,target_impact_time:targetImpactTime})});const data=await readJsonResponse(response);setResult(data);let saved=false;if(playerName.trim()){try{await storeCurrentResult(data);saved=true}catch{setError('解析は完了しましたが、端末内保存に失敗しました。ブラウザの保存容量・サイト設定を確認してください。')}}setStatus(saved?`${playerName.trim()} の結果をこの端末に保存しました。動画は削除します。`:'解析が完了しました。動画は一時保存先から削除します。');await deleteTempBlobs(urls,true)}catch(e){setError(e instanceof Error?e.message:'解析に失敗しました。');setStatus('')}}
  async function saveAgain(){if(!result)return;if(!playerName.trim()){setError('端末に保存する場合は選手名を入力してください。');return}try{await storeCurrentResult(result);setStatus(`${playerName.trim()} の結果をこの端末に保存しました。`)}catch{setError('端末内保存に失敗しました。')}}
  async function removeHistory(id:string){await deleteResultLocal(id);await refreshHistory()}
  function openSaved(item:SavedResult){setPlayerName(item.playerName);setBatterSide(item.batterSide);setResult(item.result);setStatus(`${item.playerName} の過去結果を開きました。`);window.scrollTo({top:document.body.scrollHeight*.45,behavior:'smooth'})}
  function savePdf(){window.print()}
  async function shareApp(){const url=window.location.origin;try{if(navigator.share)await navigator.share({title:'打撃フォーム分析 Visual',text:'8項目の計測箇所を画像で確認できる打撃フォーム分析です。',url});else{await navigator.clipboard.writeText(url);setStatus('共有リンクをコピーしました。')}}catch{}}

  const features=result?.features||[]; const visuals=result?.measurement_visuals||[];
  const closeCount=features.filter((r:any)=>['かなり近い','やや近い'].includes(r['比較判定'])).length;
  const diffCount=features.length-closeCount;

  return <main>
    <header className="hero no-print"><div className="version">BATTING FORM ANALYZER · VISUAL / 8 ITEMS</div><h1>打撃フォーム可視化分析</h1><p>8項目をすべて測定し、「どこを測ったのか」を実動画のコマで確認します。</p><div className="hero-actions"><button className="secondary" onClick={shareApp}>このアプリを共有</button></div></header>
    <div className="notice no-print">結果の良し悪しを断定するものではありません。比較選手との差を、単眼動画から取得できる運動学的な特徴として表示します。結果はこの端末のブラウザ内に保存できます。</div>

    <section className="step no-print"><span>STEP 1</span><h2>解析する選手</h2><div className="card">
      <label className="field"><span>選手名・ニックネーム</span><input list="known-players" value={playerName} onChange={e=>setPlayerName(e.target.value)} placeholder="例：山田 太郎（空欄なら端末保存しない）"/><datalist id="known-players">{knownPlayers.map(n=><option key={n} value={n}/>)}</datalist></label>
      <div className="segmented">{(['右打者','左打者'] as const).map(s=><button type="button" key={s} className={batterSide===s?'active':''} onClick={()=>setBatterSide(s)}>{s}</button>)}</div>
      <p className="local-note">選手名を入れて解析すると、結果・8項目の計測画像・日時をこの端末に自動保存します。動画自体は履歴に保存しません。</p>
    </div></section>

    <section className="step no-print"><span>STEP 2</span><h2>動画を選択</h2><div className="upload-grid">
      <label className="upload-card"><strong>比較選手の動画</strong><input type="file" accept="video/*" onChange={e=>{setReferenceFile(e.target.files?.[0]||null);setReferenceBlobUrl('');setReferenceRoi(DEFAULT_ROI);setReferenceImpactTime(null);setReferenceImpactPreview('');setResult(null)}}/><small>{referenceFile?.name||'2〜5秒程度がおすすめ'}</small></label>
      <label className="upload-card"><strong>自分の動画</strong><input type="file" accept="video/*" onChange={e=>{setTargetFile(e.target.files?.[0]||null);setTargetBlobUrl('');setTargetRoi(DEFAULT_ROI);setTargetImpactTime(null);setTargetImpactPreview('');setResult(null)}}/><small>{targetFile?.name||'同じ向き・近い画角がおすすめ'}</small></label>
    </div><div className="card all-features"><h3>8項目をすべて解析</h3><div className="all-feature-grid">{FEATURE_OPTIONS.map(([key,label],i)=><div key={key}><span>{i+1}</span><b>{label}</b><small>{FEATURE_HINTS[key]}</small></div>)}</div></div></section>

    <section className="step no-print"><span>STEP 3</span><h2>解析範囲とインパクト</h2><p className="step-guide">打者を赤枠で囲み、インパクトを指定します。縦動画・横動画どちらでも使えます。</p><button className="secondary full" type="button" onClick={suggestRoi} disabled={!referenceFile||!targetFile}>2本とも自動で打者位置を探す</button>
      <VideoCropCard title="比較選手" file={referenceFile} roi={referenceRoi} onChange={setReferenceRoi} impactTime={referenceImpactTime} impactPreview={referenceImpactPreview} onImpactSet={(t,p)=>{setReferenceImpactTime(t);setReferenceImpactPreview(p)}} onImpactClear={()=>{setReferenceImpactTime(null);setReferenceImpactPreview('')}}/>
      <VideoCropCard title="自分" file={targetFile} roi={targetRoi} onChange={setTargetRoi} impactTime={targetImpactTime} impactPreview={targetImpactPreview} onImpactSet={(t,p)=>{setTargetImpactTime(t);setTargetImpactPreview(p)}} onImpactClear={()=>{setTargetImpactTime(null);setTargetImpactPreview('')}}/>
    </section>

    <section className="step no-print"><span>STEP 4</span><h2>8項目を解析</h2><button className="primary full" type="button" onClick={analyze} disabled={!referenceFile||!targetFile}>8項目の高精度解析を開始</button>{(status||uploadProgress>0)&&<div className="status"><div>{status}</div>{uploadProgress>0&&uploadProgress<100&&<progress value={uploadProgress} max="100"/>}</div>}{error&&<div className="error">{error}</div>}</section>

    {result&&<section className="results" id="analysis-result">
      <div className="print-title"><h1>打撃フォーム可視化分析レポート</h1><p>{playerName||'選手名未設定'} / {new Date().toLocaleString('ja-JP')}</p></div>
      <div className="result-toolbar no-print"><button className="primary" onClick={saveAgain}>この端末に保存</button><button className="secondary" onClick={savePdf}>PDFで保存</button><button className="secondary" onClick={shareApp}>共有リンク</button></div>
      <div className="result-overview"><div className="overview-main"><span>全体傾向</span><strong>{result.overall_comparison_label}</strong><p>8項目中 <b>{closeCount}</b> 項目が比較選手に近く、<b>{diffCount}</b> 項目に差が見られました。</p></div><div className="overview-pair"><div><span>最も近い</span><b>{result.best_item}</b></div><div><span>最も違う</span><b>{result.lowest_item}</b></div></div></div>
      <div className="quality-row"><QualityBadge label={result.reference?.quality?.label}/><QualityBadge label={result.target?.quality?.label}/></div>

      <div className="result-section-head"><div><span>DETAIL</span><h2>8項目の計測画像</h2></div><p>数字だけでなく、実際にどの部位・角度・移動を使ったか確認できます。</p></div>
      <div className="measurement-grid">{FEATURE_OPTIONS.map(([key])=>{const row=features.find((r:any)=>r['評価項目']===key);const visual=visuals.find((v:any)=>v.key===key);return <MeasurementCard key={key} row={row} visual={visual}/>})}</div>

      <div className="result-section-head"><div><span>ADVICE</span><h2>優先して確認したいポイント</h2></div></div>
      <div className="advice-list">{(result.advice||[]).map((row:any,i:number)=><article className="advice" key={`${row['観点']}-${i}`}><div className="rank">{i+1}</div><div><h3>{row['観点']}</h3><p><b>比較結果：</b>{row['比較結果']}</p><p><b>動作の特徴：</b>{row['動作の特徴']}</p><p><b>改善の方向性：</b>{row['改善の方向性']}</p><p><b>練習例：</b>{row['練習例']}</p>{row['参考動画URL']&&<div className="practice-video"><div><b>日本語の参考動画</b><span>{row['参考動画提供元']}</span><strong>{row['参考動画タイトル']}</strong>{row['参考動画メモ']&&<small>{row['参考動画メモ']}</small>}</div><div className="practice-video-actions"><a href={row['参考動画URL']} target="_blank" rel="noopener noreferrer">▶ 日本語動画を見る</a>{row['予備動画URL']&&<a className="practice-fallback" href={row['予備動画URL']} target="_blank" rel="noopener noreferrer">海外版も見る</a>}</div></div>}<small>研究上の位置づけ：{row['研究上の位置づけ']}</small></div></article>)}</div>

      <div className="result-section-head"><div><span>BODY LOAD</span><h2>身体負担に関連する動作</h2></div></div><div className="body-load"><strong>{result.body_load?.overall_label}</strong><p>怪我の診断・予測ではなく、先行研究で議論される動作との照合です。</p>{(result.body_load?.regions||[]).map((row:any)=><div className="body-row" key={row['部位']}><b>{row['部位']}</b><span>{row['判定']}</span><small>{row['主な計測proxy']}</small></div>)}</div>
      <p className="report-footnote">※「体重移動」は骨盤中心の移動proxyです。真の身体重心・地面反力・筋力・関節トルクは単眼動画から直接測定していません。</p>
    </section>}

    <section className="step history-section no-print"><span>LOCAL HISTORY</span><h2>この端末の解析履歴</h2><div className="card"><p className="local-note">履歴はこのブラウザのIndexedDBに保存されます。VercelのPrivate Blobには履歴を置きません。ブラウザのサイトデータ削除・シークレットモードでは消える場合があります。</p>{historyError&&<div className="error">{historyError}</div>}{history.length===0?<div className="history-empty">まだ保存された結果はありません。</div>:<div className="history-list">{history.map(item=><div className="history-item" key={item.id}><div><b>{item.playerName}</b><time>{new Date(item.createdAt).toLocaleString('ja-JP')}</time></div><span>{item.result?.overall_comparison_label}</span><div className="history-buttons"><button onClick={()=>openSaved(item)}>結果を見る</button><button className="delete-mini" onClick={()=>removeHistory(item.id)}>削除</button></div></div>)}</div>}</div></section>
    <footer className="no-print">Batting Form Analyzer Visual · 8項目 / Measurement Overlay / Local History</footer>
  </main>;
}
