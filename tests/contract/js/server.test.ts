// The pinned JS SDK against a real gateway (fake engine). Started by tests/contract/test_server_sdk.py, which sets
// QUAN_TEST_BASE_URL, QUAN_TEST_API_KEY and QUAN_TEST_AUDIO_DIR (speech.wav, fail.wav, expected.json);
// without them (plain `npm test`) every test here is skipped.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import OpenAI, { toFile } from 'openai';
import type { TranscriptionStreamEvent } from 'openai/resources/audio/transcriptions';

const baseURL = process.env.QUAN_TEST_BASE_URL;
const apiKey = process.env.QUAN_TEST_API_KEY;
const audioDir = process.env.QUAN_TEST_AUDIO_DIR ?? '';
const skip = baseURL && apiKey && audioDir ? false : 'no gateway: run tests/contract/test_server_sdk.py';
const MODEL = 'quan-transcribe-1.0';

const examples = new URL('../examples/', import.meta.url);
const example = (name: string) => JSON.parse(readFileSync(new URL(name, examples)).toString());
const client = () => new OpenAI({ apiKey, baseURL, maxRetries: 0 });
const audio = (name: string) => toFile(readFileSync(join(audioDir, name)), name);
type Expected = { text: string; deltas: string[]; seconds: number };
const expected = (): Expected => JSON.parse(readFileSync(join(audioDir, 'expected.json')).toString());

test('models', { skip }, async () => {
  const page = await client().models.list();
  assert.deepEqual(page.data, example('models.json').data);
  assert.deepEqual(await client().models.retrieve(MODEL), example('model.json'));
  await assert.rejects(
    client().models.retrieve('whisper-1'),
    (err: unknown) => err instanceof OpenAI.NotFoundError && err.code === 'model_not_found',
  );
});

test('json and text transcription', { skip }, async () => {
  const exp = expected();
  const t = await client().audio.transcriptions.create({
    file: await audio('speech.wav'),
    model: MODEL,
    language: 'vi',
    temperature: 0,
  });
  assert.deepEqual(t, { text: exp.text, usage: { type: 'duration', seconds: exp.seconds } });
  const text = await client().audio.transcriptions.create({
    file: await audio('speech.wav'),
    model: MODEL,
    response_format: 'text',
  });
  assert.equal(text, exp.text);
});

test('SSE stream: deltas join to done', { skip }, async () => {
  const exp = expected();
  const stream = await client().audio.transcriptions.create({ file: await audio('speech.wav'), model: MODEL, stream: true });
  const events: TranscriptionStreamEvent[] = [];
  for await (const e of stream) events.push(e);
  const deltas = events.flatMap((e) => (e.type === 'transcript.text.delta' ? [e.delta] : []));
  assert.deepEqual(deltas, exp.deltas);
  const done = events.at(-1)!;
  assert.ok(done.type === 'transcript.text.done' && done.text === exp.text && done.usage === undefined);
  assert.equal(events.length, deltas.length + 1);
});

test('SSE error after headers rejects without done', { skip }, async () => {
  const stream = await client().audio.transcriptions.create({ file: await audio('fail.wav'), model: MODEL, stream: true });
  const got: string[] = [];
  await assert.rejects(
    (async () => {
      for await (const e of stream) got.push(e.type);
    })(),
    (err: unknown) => err instanceof OpenAI.APIError && err.code === 'inference_failed' && err.type === 'server_error',
  );
  assert.ok(got.length >= 1 && got.every((t) => t === 'transcript.text.delta'));
});

test('HTTP error envelope and request id', { skip }, async () => {
  await assert.rejects(
    client().audio.transcriptions.create({ file: await audio('speech.wav'), model: MODEL, language: 'en' }),
    (err: unknown) =>
      err instanceof OpenAI.BadRequestError &&
      err.code === 'unsupported_language' &&
      err.param === 'language' &&
      (err.requestID ?? '').startsWith('req_'),
  );
});
