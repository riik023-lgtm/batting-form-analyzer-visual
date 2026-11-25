import { del } from '@vercel/blob';

export async function POST(request: Request): Promise<Response> {
  try {
    const body = (await request.json()) as { urls?: string[] };
    const urls = (body.urls ?? []).filter(Boolean);
    if (urls.length) await del(urls);
    return Response.json({ ok: true });
  } catch (error) {
    return Response.json({ ok: false, error: error instanceof Error ? error.message : 'delete error' }, { status: 400 });
  }
}
