// The examples in ../examples (docs/api-contract.vi.md) against the pinned OpenAI JS SDK.
// `tsc --noEmit` type-checks the typed copies below; `node --test` runs this file (Node >= 23.6 strips types),
// checks the typed copies equal the shared example files, and parses the HTTP examples with the real SDK client.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import OpenAI, { toFile } from 'openai';
import type { Model } from 'openai/resources/models';
import type { Transcription, TranscriptionStreamEvent } from 'openai/resources/audio/transcriptions';
import type { RealtimeClientEvent, RealtimeServerEvent } from 'openai/resources/realtime/realtime';

const dir = new URL('../examples/', import.meta.url);
const read = (name: string) => readFileSync(new URL(name, dir));
const jsonl = (name: string) =>
  read(name).toString('utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
const MODEL = 'quan-transcribe-1.0';

const model = { id: MODEL, object: 'model', created: 1791950390, owned_by: 'qnguyen3' } satisfies Model;

const session = {
  type: 'transcription',
  audio: {
    input: {
      format: { type: 'audio/pcm', rate: 24000 },
      transcription: { model: MODEL, language: 'vi' },
      turn_detection: null,
    },
  },
} as const;

const clientEvents = [
  {
    type: 'session.update',
    event_id: 'client_evt_1',
    session: {
      type: 'transcription',
      audio: {
        input: {
          format: { type: 'audio/pcm', rate: 24000 },
          transcription: { model: MODEL, language: 'vi' },
          turn_detection: null,
        },
      },
    },
  },
  { type: 'input_audio_buffer.append', event_id: 'client_evt_2', audio: 'AAABAP//AAA=' },
  { type: 'input_audio_buffer.commit', event_id: 'client_evt_3' },
  { type: 'input_audio_buffer.clear', event_id: 'client_evt_4' },
] satisfies RealtimeClientEvent[];

const serverEvents = [
  { type: 'session.created', event_id: 'event_0001', session },
  { type: 'session.updated', event_id: 'event_0002', session },
  { type: 'input_audio_buffer.committed', event_id: 'event_0003', item_id: 'item_0001', previous_item_id: null },
  {
    type: 'conversation.item.added',
    event_id: 'event_0004',
    previous_item_id: null,
    item: {
      id: 'item_0001',
      object: 'realtime.item',
      type: 'message',
      role: 'user',
      status: 'completed',
      content: [{ type: 'input_audio' }],
    },
  },
  {
    type: 'conversation.item.input_audio_transcription.delta',
    event_id: 'event_0005',
    item_id: 'item_0001',
    content_index: 0,
    delta: 'một số loài động vật như voi',
  },
  {
    type: 'conversation.item.input_audio_transcription.delta',
    event_id: 'event_0006',
    item_id: 'item_0001',
    content_index: 0,
    delta: ' và hươu cao cổ',
  },
  {
    type: 'conversation.item.input_audio_transcription.completed',
    event_id: 'event_0007',
    item_id: 'item_0001',
    content_index: 0,
    transcript: 'một số loài động vật như voi và hươu cao cổ',
    usage: { type: 'duration', seconds: 4.2 },
  },
  { type: 'input_audio_buffer.cleared', event_id: 'event_0008' },
  {
    type: 'error',
    event_id: 'event_0009',
    error: {
      type: 'invalid_request_error',
      code: 'invalid_audio',
      message: 'input_audio_buffer.append: audio is not valid base64.',
      param: 'audio',
      event_id: 'client_evt_5',
    },
  },
  {
    type: 'conversation.item.input_audio_transcription.failed',
    event_id: 'event_0010',
    item_id: 'item_0002',
    content_index: 0,
    error: { type: 'server_error', code: 'inference_failed', message: 'Transcription failed for this item.' },
  },
] satisfies RealtimeServerEvent[];

test('typed copies equal the shared example files', () => {
  assert.deepEqual(model, JSON.parse(read('model.json').toString()));
  assert.deepEqual(clientEvents, jsonl('realtime_client.jsonl'));
  assert.deepEqual(serverEvents, jsonl('realtime_server.jsonl'));
  const t: Transcription = JSON.parse(read('transcription.json').toString());
  assert.equal(t.usage?.type, 'duration');
});

// What the SDK handed to fetch(); multipart bodies arrive as a native FormData.
type Sent = { url: string; headers: Headers; body: unknown };

function client(status: number, body: Buffer, contentType: string, seen: Sent[] = []) {
  return new OpenAI({
    apiKey: 'sk-test',
    baseURL: 'http://gateway.test/v1',
    maxRetries: 0,
    fetch: async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input instanceof Request ? input.url : input);
      // The SDK first probes fetch('data:,') for FormData support; record gateway requests only.
      if (url.startsWith('http://gateway.test/')) seen.push({ url, headers: new Headers(init?.headers), body: init?.body });
      return new Response(new Uint8Array(body), { status, headers: { 'content-type': contentType } });
    },
  });
}

const audio = () => toFile(readFileSync(new URL('../../fixtures/vi_a.wav', import.meta.url)), 'vi_a.wav');

test('models', async () => {
  const seen: Sent[] = [];
  const page = await client(200, read('models.json'), 'application/json', seen).models.list();
  assert.deepEqual(page.data.map((m) => m.id), [MODEL]);
  assert.equal(new URL(seen[0]!.url).pathname, '/v1/models');
  assert.equal(seen[0]!.headers.get('authorization'), 'Bearer sk-test');
});

test('json and text transcription', async () => {
  const seen: Sent[] = [];
  const c = client(200, read('transcription.json'), 'application/json', seen);
  const t = await c.audio.transcriptions.create({ file: await audio(), model: MODEL, language: 'vi' });
  assert.ok(t.text.startsWith('các sản phẩm'));
  const form = await new Response(seen[0]!.body as BodyInit).formData(); // encode + parse as multipart
  assert.deepEqual([...form.keys()].sort(), ['file', 'language', 'model']);

  const text = await client(200, read('transcription.txt'), 'text/plain; charset=utf-8').audio.transcriptions.create(
    { file: await audio(), model: MODEL, response_format: 'text' },
  );
  assert.equal(text, read('transcription.txt').toString('utf8'));
});

test('array fields are sent as repeated bracket fields', async () => {
  const seen: Sent[] = [];
  await client(200, read('transcription.json'), 'application/json', seen).audio.transcriptions.create({
    file: await audio(),
    model: MODEL,
    response_format: 'verbose_json',
    timestamp_granularities: ['word', 'segment'],
    include: ['logprobs'],
  });
  const form = await new Response(seen[0]!.body as BodyInit).formData(); // encode + parse as multipart
  assert.deepEqual(form.getAll('timestamp_granularities[]'), ['word', 'segment']);
  assert.deepEqual(form.getAll('include[]'), ['logprobs']);
});

test('SSE stream: deltas join to done', async () => {
  const stream = await client(200, read('transcription_stream.sse'), 'text/event-stream').audio.transcriptions.create({
    file: await audio(),
    model: MODEL,
    stream: true,
  });
  const events: TranscriptionStreamEvent[] = [];
  for await (const e of stream) events.push(e);
  assert.deepEqual(events.map((e) => e.type), ['transcript.text.delta', 'transcript.text.delta', 'transcript.text.done']);
  const done = events.at(-1)!;
  assert.ok(done.type === 'transcript.text.done');
  const joined = events.map((e) => (e.type === 'transcript.text.delta' ? e.delta : '')).join('');
  assert.equal(joined, done.text);
});

test('SSE error after headers rejects without done', async () => {
  const stream = await client(200, read('transcription_stream_error.sse'), 'text/event-stream').audio.transcriptions.create(
    { file: await audio(), model: MODEL, stream: true },
  );
  const got: string[] = [];
  await assert.rejects(
    (async () => {
      for await (const e of stream) got.push(e.type);
    })(),
    (err: unknown) => err instanceof OpenAI.APIError && err.code === 'inference_failed',
  );
  assert.deepEqual(got, ['transcript.text.delta']);
});

test('HTTP error envelope', async () => {
  await assert.rejects(
    client(400, read('error.json'), 'application/json').audio.transcriptions.create({
      file: await audio(),
      model: MODEL,
      language: 'en',
    }),
    (err: unknown) =>
      err instanceof OpenAI.BadRequestError && err.code === 'unsupported_language' && err.param === 'language',
  );
});
