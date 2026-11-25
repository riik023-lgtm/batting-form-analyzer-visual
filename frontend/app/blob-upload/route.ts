import { setTtl } from '@vercel/blob';
import { handleUpload, type HandleUploadBody } from '@vercel/blob/client';

export async function POST(request: Request): Promise<Response> {
  const body = (await request.json()) as HandleUploadBody;
  try {
    const response = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async () => ({
        allowedContentTypes: [
          'video/mp4',
          'video/quicktime',
          'video/x-m4v',
          'video/x-msvideo',
          'video/webm',
          'application/octet-stream'
        ],
        maximumSizeInBytes: 250 * 1024 * 1024,
        addRandomSuffix: true,
      }),
      onUploadCompleted: async ({ blob }) => {
        try { await setTtl(blob.url, { ttlDays: 1 }); } catch {}
      },
    });
    return Response.json(response);
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : 'upload error' }, { status: 400 });
  }
}
