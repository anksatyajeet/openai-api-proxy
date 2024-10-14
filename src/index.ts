import { Hono } from 'hono';
import { streamSSE } from 'hono/streaming';
import { openai } from './llm/openai';
import { anthropic, anthropicVertex } from './llm/anthropic';
import OpenAI from 'openai';
import { google } from './llm/google';
import { deepseek } from './llm/deepseek';
import { serializeError } from 'serialize-error';
import { HTTPException } from 'hono/http-exception';
import { cors } from 'hono/cors';
import { moonshot } from './llm/moonshot';
import { lingyiwanwu } from './llm/lingyiwanwu';
import { groq } from './llm/groq';
import { auzreOpenAI } from './llm/azure';
import { bailian } from './llm/bailian';
import { cohere } from './llm/cohere';

interface Bindings {
  API_KEY: string;
  OPENAI_API_KEY: string;
}

function getModels(env: Record<string, string>) {
  return [
    openai(env),
    anthropic(env),
    anthropicVertex(env),
    google(env),
    deepseek(env),
    moonshot(env),
    lingyiwanwu(env),
    groq(env),
    auzreOpenAI(env),
    cohere(env),
    bailian(env),
  ].filter((it) => it.requiredEnv.every((it) => it in env));
}

const app = new Hono<{ Bindings: Bindings }>()
  .use(
    cors({
      origin: (_origin, c) => {
        return c.env.CORS_ORIGIN;
      },
    }),
  )
  .use(async (c, next) => {
    await next();
    if (c.error) {
      throw new HTTPException((c.error as any)?.status ?? 500, {
        message: serializeError(c.error).message,
      });
    }
  })
  .options('/v1/chat/completions', async (c) => {
    return c.json({ body: 'ok' });
  })
  .use(async (c, next) => {
    const authHeader = c.req.header('Authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    const token = authHeader.split(' ')[1];
    if (token !== c.env.API_KEY) {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    return next();
  })
  .post('/v1/chat/completions', async (c) => {
    const req = (await c.req.json()) as
      | OpenAI.ChatCompletionCreateParamsNonStreaming
      | OpenAI.ChatCompletionCreateParamsStreaming;

    const list = getModels(c.env as any);
    const llm = list.find((it) => it.supportModels.includes(req.model));
    if (!llm) {
      return c.json({ error: `Model ${req.model} not supported` }, 400);
    }

    // Pass x-api-key to the AI API
    const apiKey = c.req.header('x-api-key');

    if (req.stream) {
      const abortController = new AbortController();
      return streamSSE(c, async (stream) => {
        stream.onAbort(() => abortController.abort());
        for await (const it of llm.stream(req, abortController.signal, apiKey)) {
          stream.writeSSE({ data: JSON.stringify(it) });
        }
      });
    }

    return c.json(await llm?.invoke(req, apiKey));
  })
  .get('/v1/models', async (c) => {
    return c.json({
      object: 'list',
      data: getModels(c.env as any).flatMap((it) =>
        it.supportModels.map(
          (model) =>
            ({
              id: model,
              object: 'model',
              owned_by: it.name,
              created: Math.floor(Date.now() / 1000),
            } as OpenAI.Models.Model),
        ),
      ),
    } as OpenAI.Models.ModelsPage);
  });

export default app;
