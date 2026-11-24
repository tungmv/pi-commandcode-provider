import assert from "node:assert/strict";
import test from "node:test";

import registerCommandCode from "../index.ts";

let provider;
registerCommandCode({
	registerProvider(name, config) {
		assert.equal(name, "commandcode");
		provider = config;
	},
});

const model = {
	id: "deepseek/deepseek-v4-flash",
	name: "DeepSeek V4 Flash",
	api: "commandcode-generate",
	provider: "commandcode",
	reasoning: true,
	input: ["text"],
	contextWindow: 1_000_000,
	maxTokens: 64_000,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

test("model registry includes the current OSS roster and vision capabilities", () => {
	const byID = new Map(provider.models.map((entry) => [entry.id, entry]));
	for (const id of [
		"deepseek/deepseek-v4.1-flash",
		"deepseek/deepseek-v4.1-flash-fast",
		"deepseek/deepseek-v4.1-flash-ultrafast",
		"moonshotai/Kimi-K3",
		"zai-org/GLM-5.2-Fast",
		"tencent/Hy3",
		"thinkingmachines/inkling",
	]) {
		assert.ok(byID.has(id), `missing current model ${id}`);
	}
	for (const id of [
		"moonshotai/Kimi-K3",
		"moonshotai/Kimi-K2.7-Code",
		"moonshotai/Kimi-K2.7-Code-Highspeed",
		"moonshotai/Kimi-K2.6",
		"moonshotai/Kimi-K2.5",
		"MiniMaxAI/MiniMax-M3",
		"Qwen/Qwen3.7-Plus",
		"stepfun/Step-3.7-Flash",
		"thinkingmachines/inkling",
	]) {
		assert.ok(byID.get(id)?.input.includes("image"), `${id} should accept image input`);
	}
});

test("truncated NDJSON stream fails instead of emitting done", async (t) => {
	t.mock.method(globalThis, "fetch", async () => {
		return new Response(
			[
				'{"type":"start"}',
				'{"type":"text-start","id":"txt-0"}',
				'{"type":"text-delta","id":"txt-0","text":"partial"}',
			].join("\n"),
			{ status: 200, headers: { "content-type": "application/x-ndjson" } },
		);
	});

	const events = [];
	const stream = provider.streamSimple(model, { systemPrompt: "", messages: [], tools: [] }, { apiKey: "user_test" });
	for await (const event of stream) {
		events.push(event);
	}

	assert.equal(events.at(-1)?.type, "error");
	assert.match(events.at(-1)?.error?.errorMessage ?? "", /terminal event/i);
});

test("tool-call accepts the current args fallback", async (t) => {
	t.mock.method(globalThis, "fetch", async (_input, init) => {
		assert.equal(init.headers["x-command-code-version"], "0.52.1");
		return new Response(
			[
				'{"type":"start"}',
				'{"type":"tool-call","toolCallId":"call_1","toolName":"get_weather","args":{"location":"Paris"}}',
				'{"type":"finish","finishReason":"tool-calls","totalUsage":{"inputTokens":10,"outputTokens":3,"totalTokens":13}}',
			].join("\n"),
			{ status: 200, headers: { "content-type": "application/x-ndjson" } },
		);
	});

	const events = [];
	const stream = provider.streamSimple(model, { systemPrompt: "", messages: [], tools: [] }, { apiKey: "user_test" });
	for await (const event of stream) {
		events.push(event);
	}

	const toolEnd = events.find((event) => event.type === "toolcall_end");
	assert.deepEqual(toolEnd?.toolCall?.arguments, { location: "Paris" });
	assert.equal(events.at(-1)?.message?.usage?.output, 3);
});

test("CMD_ZDR enables the strict zero-retention header", async (t) => {
	const previous = process.env.CMD_ZDR;
	process.env.CMD_ZDR = "1";
	t.after(() => {
		if (previous === undefined) delete process.env.CMD_ZDR;
		else process.env.CMD_ZDR = previous;
	});

	let zdrHeader;
	t.mock.method(globalThis, "fetch", async (_input, init) => {
		zdrHeader = init.headers["x-cmd-zdr"];
		return new Response('{"type":"finish","finishReason":"stop","totalUsage":{"totalTokens":0}}\n', {
			status: 200,
			headers: { "content-type": "application/x-ndjson" },
		});
	});

	const events = [];
	const stream = provider.streamSimple(model, { systemPrompt: "", messages: [], tools: [] }, { apiKey: "user_test" });
	for await (const event of stream) events.push(event);

	assert.equal(zdrHeader, "1");
	assert.equal(events.at(-1)?.type, "done");
});

// ---- Text-mode tool-call recovery ---------------------------------------
// A model answering with tool markup in the text channel, as captured from
// DeepSeek v4/v4.1 through pi: the gateway emits only text events and
// finishReason "stop", so the call would otherwise vanish.

function textModeStream(text) {
	return [
		'{"type":"start"}',
		'{"type":"reasoning-start","id":"reasoning-0"}',
		'{"type":"reasoning-delta","id":"reasoning-0","text":"thinking"}',
		'{"type":"reasoning-end","id":"reasoning-0"}',
		'{"type":"text-start","id":"txt-0"}',
		JSON.stringify({ type: "text-delta", id: "txt-0", text }),
		'{"type":"text-end","id":"txt-0"}',
		'{"type":"finish-step","finishReason":"stop","usage":{"inputTokens":10,"outputTokens":5}}',
	].join("\n");
}

async function runStream(t, body) {
	// The stream must be iterated only after fetch is mocked, or the extension
	// escapes into the real gateway.
	t.mock.method(globalThis, "fetch", async () => {
		return new Response(body, { status: 200, headers: { "content-type": "application/x-ndjson" } });
	});

	const events = [];
	const stream = provider.streamSimple(
		model,
		{ systemPrompt: "", messages: [], tools: [] },
		{ apiKey: "user_test" },
	);
	for await (const event of stream) events.push(event);
	return events;
}

test("recovers a text-mode DSML tool call instead of reporting a plain stop", async (t) => {
	const events = await runStream(t,
		textModeStream(
			'Reading the file now.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n' +
				'<｜DSML｜ parameter name="command" string="true">echo "hi"; cat -n ~/.config/hypr/input.lua</｜DSML｜ parameter>\n' +
				'<｜DSML｜ parameter name="timeout" string="false">10</｜DSML｜ parameter>\n' +
				'</｜DSML｜ invoke>\n</｜DSML｜ calls>',
		),
	);

	const toolEnds = events.filter((e) => e.type === "toolcall_end");
	assert.equal(toolEnds.length, 1);
	assert.equal(toolEnds[0].toolCall.name, "bash");
	assert.deepEqual(toolEnds[0].toolCall.arguments, {
		command: 'echo "hi"; cat -n ~/.config/hypr/input.lua',
		timeout: 10,
	});

	const done = events.at(-1);
	assert.equal(done.type, "done");
	assert.equal(done.reason, "toolUse");
	assert.equal(done.message.stopReason, "toolUse");
	// The markup must not survive as visible text.
	assert.ok(!JSON.stringify(done.message).includes("DSML"));
	// Prose before the markup is kept, ahead of the recovered call.
	const textBlock = done.message.content.find((b) => b.type === "text");
	assert.match(textBlock.text, /Reading the file now/);
	assert.ok(
		done.message.content.findIndex((b) => b.type === "text") <
			done.message.content.findIndex((b) => b.type === "toolCall"),
	);
});

test("recovers tool calls from the other observed bracket spellings", async (t) => {
	for (const text of [
		'<||DSML|| calls><||DSML|| invoke name="bash"><||DSML|| parameter name="command">ls</||DSML|| parameter></||DSML|| invoke></||DSML|| calls>',
		'<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="bash"><｜｜DSML｜｜ parameter name="command">ls</｜｜DSML｜｜ parameter></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>',
		'<|DSML|><|DSML| invoke name="bash"><|DSML| parameter name="command">ls</|DSML| parameter></|DSML| invoke></|DSML|>',
	]) {
		const events = await runStream(t, textModeStream(text));
		const toolEnds = events.filter((e) => e.type === "toolcall_end");
		assert.equal(toolEnds.length, 1, `failed to recover from: ${text}`);
		assert.equal(toolEnds[0].toolCall.name, "bash");
		assert.deepEqual(toolEnds[0].toolCall.arguments, { command: "ls" });
	}
});

test("recovers several invokes from one text block", async (t) => {
	const events = await runStream(t,
		textModeStream(
			'<｜DSML｜ calls>\n' +
				'<｜DSML｜ invoke name="bash"><｜DSML｜ parameter name="command" string="true">pwd</｜DSML｜ parameter></｜DSML｜ invoke>\n' +
				'<｜DSML｜ invoke name="bash"><｜DSML｜ parameter name="command" string="true">ls -la</｜DSML｜ parameter></｜DSML｜ invoke>\n' +
				'</｜DSML｜ calls>',
		),
	);

	const toolEnds = events.filter((e) => e.type === "toolcall_end");
	assert.equal(toolEnds.length, 2);
	assert.deepEqual(
		toolEnds.map((e) => e.toolCall.arguments.command),
		["pwd", "ls -la"],
	);
});

test("parses JSON-valued and escaped parameters", async (t) => {
	const events = await runStream(t,
		textModeStream(
			'<｜DSML｜ invoke name="edit">' +
				'<｜DSML｜ parameter name="path">/tmp/a.lua</｜DSML｜ parameter>' +
				'<｜DSML｜ parameter name="replacements">{"oldText":"a","newText":"b"}</｜DSML｜ parameter>' +
				'</｜DSML｜ invoke>',
		),
	);

	const toolEnd = events.find((e) => e.type === "toolcall_end");
	assert.equal(toolEnd.toolCall.name, "edit");
	assert.deepEqual(toolEnd.toolCall.arguments, {
		path: "/tmp/a.lua",
		replacements: { oldText: "a", newText: "b" },
	});
});

test("structured tool calls still win and text recovery stays out of the way", async (t) => {
	const events = await runStream(t,
		[
			'{"type":"start"}',
			'{"type":"text-start","id":"txt-0"}',
			'{"type":"text-delta","id":"txt-0","text":"Calling a tool."}',
			'{"type":"text-end","id":"txt-0"}',
			'{"type":"tool-input-start","id":"call_1","toolName":"read_file"}',
			'{"type":"tool-input-delta","id":"call_1","delta":"{\\"path\\":\\"/etc/hostname\\"}"}',
			'{"type":"tool-input-end","id":"call_1"}',
			'{"type":"tool-call","toolCallId":"call_1","toolName":"read_file","input":{"path":"/etc/hostname"}}',
			'{"type":"finish-step","finishReason":"tool-calls"}',
		].join("\n"),
	);

	const toolEnds = events.filter((e) => e.type === "toolcall_end");
	assert.equal(toolEnds.length, 1);
	assert.equal(toolEnds[0].toolCall.id, "call_1");
	assert.deepEqual(toolEnds[0].toolCall.arguments, { path: "/etc/hostname" });
});

test("ordinary prose is untouched by text recovery", async (t) => {
	const events = await runStream(t,
		textModeStream("The file `/etc/hostname` contains `omax`, the system's hostname."),
	);

	const toolEnds = events.filter((e) => e.type === "toolcall_end");
	assert.equal(toolEnds.length, 0);
	const done = events.at(-1);
	assert.equal(done.reason, "stop");
	const textBlock = done.message.content.find((b) => b.type === "text");
	assert.match(textBlock.text, /contains `omax`/);
});

test("recovers a Hermes-style function envelope", async (t) => {
	const events = await runStream(
		t,
		textModeStream('I will run it now.\n\n' + "<tool_call>\n<function name=\"bash\">\n<parameter name=\"command\">echo HELLO_FROM_TOOL</parameter>\n<parameter name=\"timeout\">10</parameter>\n</function>\n</tool_call>"),
	);

	const toolEnds = events.filter((e) => e.type === "toolcall_end");
	assert.equal(toolEnds.length, 1);
	assert.equal(toolEnds[0].toolCall.name, "bash");
	assert.deepEqual(toolEnds[0].toolCall.arguments, {
		command: "echo HELLO_FROM_TOOL",
		timeout: 10,
	});

	const done = events.at(-1);
	assert.equal(done.reason, "toolUse");
	assert.ok(!JSON.stringify(done.message).includes("tool_call"));
});
