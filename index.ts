/**
 * Command Code provider extension for pi.
 *
 * Routes pi requests through https://api.commandcode.ai/alpha/generate
 * (the CLI's own endpoint), which every plan permits. The /provider/v1/*
 * OpenAI/Anthropic-compatible endpoints have been flaky/plan-gated for
 * generation, so we stick to the native /alpha/generate envelope the CLI
 * itself uses (GET /provider/v1/models is readable for the roster).
 *
 * Authenticates with COMMANDCODE_API_KEY (a "user_..." token from
 * https://commandcode.ai/settings/billing or `cmd auth status`).
 *
 * Wire shape, sender:
 *   POST /alpha/generate
 *   body: { config, memory: "", taste: null, skills: null,
 *           permissionMode: "standard",
 *           params: { model, system, messages, tools, max_tokens, stream } }
 *
 * Wire shape, receiver (newline-delimited JSON, NOT SSE):
 *   {"type":"start"}
 *   {"type":"start-step",...}
 *   {"type":"reasoning-start","id":"reasoning-0"}
 *   {"type":"reasoning-delta","id":"reasoning-0","text":"..."}
 *   {"type":"reasoning-end","id":"reasoning-0"}
 *   {"type":"text-start","id":"txt-0"}
 *   {"type":"text-delta","id":"txt-0","text":"..."}
 *   {"type":"text-end","id":"txt-0"}
 *   {"type":"tool-input-start","id":"call_...","toolName":"..."}
 *   {"type":"tool-input-delta","id":"call_...","delta":"<json chunk>"}
 *   {"type":"tool-input-end","id":"call_..."}
 *   {"type":"finish-step","finishReason":"stop"|"length"|"tool-calls","usage":{...}}
 */

import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	type Message,
	type JsonObject,
	type Model,
	type Tool,
	type SimpleStreamOptions,
	calculateCost,
	createAssistantMessageEventStream,
	getCurrentSystemPrompt,
	getCurrentTools,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE_URL = "https://api.commandcode.ai";
const ENDPOINT = "/alpha/generate";
// Mirrors the `x-command-code-version` the CLI sends (its package version).
// Keep in step with the installed `command-code` package; stale values risk
// being rejected or flagged by the gateway.
const COMMAND_CODE_VERSION = "0.52.1";

// ---- Models -------------------------------------------------------------
// IDs are the gateway's canonical ids from GET /provider/v1/models
// (readable with a user_... key as of CLI 0.52.x).
//
// Only the open-weight / OSS roster is exposed here. Command Code also serves
// proprietary frontier models (claude-*, gpt-*, google/gemini-*) through the
// same /alpha/generate envelope, but they bill real plan credits and are
// available far more cheaply elsewhere, so they're intentionally omitted.
// To add one, append a ModelDef with the canonical id from the models list.
//
// The Go plan ($1/mo, $10 credits) has usage multipliers on some OSS models
// (e.g. mimo-v2.5 is ~10x, mimo-v2.5-pro ~5x, deepseek-v4-pro ~4x,
// Qwen3.7-Max ~2x) — heavy ones burn credits fast.
//
// maxTokens here is the per-call output cap pi sends as max_tokens; the gateway
// clamps anything larger to the model's true limit.

type ModelDef = {
	id: string;
	name: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	contextWindow: number;
	maxTokens: number;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
};

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

const MODELS: ModelDef[] = [
	// DeepSeek
	{
		id: "deepseek/deepseek-v4.1-flash",
		name: "DeepSeek V4.1 Flash (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "deepseek/deepseek-v4.1-flash-fast",
		name: "DeepSeek V4.1 Flash Fast (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "deepseek/deepseek-v4.1-flash-ultrafast",
		name: "DeepSeek V4.1 Flash Ultrafast (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_048_576,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "deepseek/deepseek-v4-pro",
		name: "DeepSeek V4 Pro (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "deepseek/deepseek-v4-flash",
		name: "DeepSeek V4 Flash (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	// Moonshot Kimi
	{
		id: "moonshotai/Kimi-K3",
		name: "Kimi K3 (Command Code)",
		reasoning: true,
		input: ["text", "image"],
		contextWindow: 1_000_000,
		maxTokens: 65_536,
		cost: ZERO_COST,
	},
	{
		id: "moonshotai/Kimi-K2.7-Code",
		name: "Kimi K2.7 Code (Command Code)",
		reasoning: true,
		input: ["text", "image"],
		contextWindow: 256000,
		maxTokens: 65536,
		cost: ZERO_COST,
	},
	{
		id: "moonshotai/Kimi-K2.7-Code-Highspeed",
		name: "Kimi K2.7 Code HighSpeed (Command Code)",
		reasoning: true,
		input: ["text", "image"],
		contextWindow: 262000,
		maxTokens: 65536,
		cost: ZERO_COST,
	},
	{
		id: "moonshotai/Kimi-K2.6",
		name: "Kimi K2.6 (Command Code)",
		reasoning: false,
		input: ["text", "image"],
		contextWindow: 256000,
		maxTokens: 65536,
		cost: ZERO_COST,
	},
	{
		id: "moonshotai/Kimi-K2.5",
		name: "Kimi K2.5 (Command Code)",
		reasoning: false,
		input: ["text", "image"],
		contextWindow: 256000,
		maxTokens: 65536,
		cost: ZERO_COST,
	},
	// Zhipu GLM
	{
		id: "zai-org/GLM-5.2",
		name: "GLM 5.2 (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "zai-org/GLM-5.2-Fast",
		name: "GLM 5.2 Fast (Command Code)",
		reasoning: false,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 65_536,
		cost: ZERO_COST,
	},
	{
		id: "zai-org/GLM-5.1",
		name: "GLM 5.1 (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 200000,
		maxTokens: 32768,
		cost: ZERO_COST,
	},
	{
		id: "zai-org/GLM-5",
		name: "GLM 5 (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 200000,
		maxTokens: 32768,
		cost: ZERO_COST,
	},
	// MiniMax
	{
		id: "MiniMaxAI/MiniMax-M3",
		name: "MiniMax M3 (Command Code)",
		reasoning: true,
		input: ["text", "image"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "MiniMaxAI/MiniMax-M2.7",
		name: "MiniMax M2.7 (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 200000,
		maxTokens: 65536,
		cost: ZERO_COST,
	},
	{
		id: "MiniMaxAI/MiniMax-M2.5",
		name: "MiniMax M2.5 (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 200000,
		maxTokens: 65536,
		cost: ZERO_COST,
	},
	// Xiaomi MiMo
	{
		id: "xiaomi/mimo-v2.5-pro",
		name: "MiMo V2.5 Pro (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "xiaomi/mimo-v2.5",
		name: "MiMo V2.5 (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	// Qwen
	{
		id: "Qwen/Qwen3.7-Max",
		name: "Qwen 3.7 Max (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "Qwen/Qwen3.7-Plus",
		name: "Qwen 3.7 Plus (Command Code)",
		reasoning: true,
		input: ["text", "image"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	{
		id: "Qwen/Qwen3.6-Max-Preview",
		name: "Qwen 3.6 Max Preview (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 200000,
		maxTokens: 32768,
		cost: ZERO_COST,
	},
	{
		id: "Qwen/Qwen3.6-Plus",
		name: "Qwen 3.6 Plus (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 200000,
		maxTokens: 32768,
		cost: ZERO_COST,
	},
	// StepFun
	{
		id: "stepfun/Step-3.7-Flash",
		name: "Step 3.7 Flash (Command Code)",
		reasoning: true,
		input: ["text", "image"],
		contextWindow: 256000,
		maxTokens: 65536,
		cost: ZERO_COST,
	},
	{
		id: "stepfun/Step-3.5-Flash",
		name: "Step 3.5 Flash (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	// Tencent
	{
		id: "tencent/Hy3",
		name: "Tencent Hy3 (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 262_144,
		maxTokens: 65_536,
		cost: ZERO_COST,
	},
	// NVIDIA
	{
		id: "nvidia/nemotron-3-ultra-550b-a55b",
		name: "Nemotron 3 Ultra (Command Code)",
		reasoning: true,
		input: ["text"],
		contextWindow: 1_000_000,
		maxTokens: 131072,
		cost: ZERO_COST,
	},
	// Thinking Machines
	{
		id: "thinkingmachines/inkling",
		name: "Inkling (Command Code)",
		reasoning: true,
		input: ["text", "image"],
		contextWindow: 256_000,
		maxTokens: 65_536,
		cost: ZERO_COST,
	},
];

// ---- Message conversion -------------------------------------------------
//
// The gateway speaks Vercel AI SDK's ModelMessage schema (not Anthropic's
// content-block shape). Verified by reading the CLI bundle's own converter
// and by hitting the gateway with both shapes — the Anthropic shape errors
// with "messages do not match the ModelMessage[] schema."
//
// Shape:
//   user:      { role: "user",      content: [{ type: "text", text }, { type: "image", image, mediaType }] }
//   assistant: { role: "assistant", content: [{ type: "text", text }, { type: "tool-call", toolCallId, toolName, input }] }
//   tool:      { role: "tool",      content: [{ type: "tool-result", toolCallId, toolName, output: { type: "text", value } }] }

type UserContent =
	| { type: "text"; text: string }
	| { type: "image"; image: string; mediaType?: string };

type AssistantContent =
	| { type: "text"; text: string }
	| { type: "tool-call"; toolCallId: string; toolName: string; input: Record<string, unknown> };

type ToolContent = {
	type: "tool-result";
	toolCallId: string;
	toolName: string;
	output: { type: "text"; value: string } | { type: "error-text"; value: string };
};

type CmdMessage =
	| { role: "user"; content: UserContent[] }
	| { role: "assistant"; content: AssistantContent[] }
	| { role: "tool"; content: ToolContent[] };

function stringifyUnknown(value: unknown): string {
	if (value instanceof Error) return value.message || value.stack || "Error";
	if (typeof value === "string") return value;
	try {
		return JSON.stringify(value) ?? String(value);
	} catch {
		return "Unknown error (non-serializable)";
	}
}

function parseJsonLine(line: string, lineNumber: number): unknown {
	try {
		return JSON.parse(line);
	} catch (error) {
		const preview = line.length > 240 ? `${line.slice(0, 240)}...` : line;
		throw new Error(
			`Malformed Command Code NDJSON at line ${lineNumber}: ${preview} (${stringifyUnknown(error)})`,
		);
	}
}

// ---- Text-mode tool-call recovery ---------------------------------------
// Some models routed through /alpha/generate answer with tool-call markup in
// the *text* channel instead of emitting `tool-input-*` events. The gateway
// reports finishReason "stop" for those, so without this the call is silently
// dropped: the turn looks like a normal answer and the agent loop stalls.
//
// Two markup families have been observed from DeepSeek v4/v4.1 behind pi's
// real prompt (long system prompt, ~15 tools, a long thinking block):
//
//   1. DSML-ish, pipe-decorated with a tag name:
//
//        [PIPE]DSML[PIPE] calls
//        [PIPE]DSML[PIPE] invoke name="bash"
//        [PIPE]DSML[PIPE] parameter name="command" string="true">...
//        [PIPE]DSML[PIPE] /invoke
//
//   2. An OpenAI/Hermes-style function envelope:
//
//        [LT]tool_call[GT]
//        [LT]function name="bash"[GT]
//        [LT]parameter name="command"[GT]...[/parameter]
//        [/function]
//        [/tool_call]
//
// The pipe decoration is not one fixed spelling: `[PIPE][PIPE]DSML[PIPE][PIPE]`,
// `[PIPE]|DSML|[PIPE]`, and a bare tag have all been seen. So the patterns
// below take an optional run of pipe-ish characters plus an optional tag name
// on either side of the element name, and the function envelope is tried as a
// second shape. Values may be JSON, a JSON string literal, or bare text.

const PIPE = "[|｜]";
const TAG_NAME = "[A-Za-z_][\\w.-]*";
const TAG_PREFIX = `(?:${PIPE}{1,4}${TAG_NAME}${PIPE}{1,4}|${PIPE}{1,4})?`;

// DSML family: an invoke element wrapping parameter elements.
const INVOKE_RE = new RegExp(
	`<\\s*${TAG_PREFIX}\\s*invoke\\b([^>]*)>([\\s\\S]*?)<\\s*\\/\\s*${TAG_PREFIX}\\s*invoke\\s*>`,
	"gi",
);
// Function envelope: an element whose name is the tool, wrapping parameter elements.
const FUNCTION_RE =
	/<\s*function\b([^>]*)>([\s\S]*?)<\s*\/\s*function\s*>/gi;
const PARAMETER_RE = new RegExp(
	`<\\s*${TAG_PREFIX}\\s*parameter\\b([^>]*)>([\\s\\S]*?)<\\s*\\/\\s*${TAG_PREFIX}\\s*parameter\\s*>`,
	"gi",
);
const NAME_ATTR_RE = /\bname\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i;
// Closing tag for the envelope form, used to strip leftovers.
const STRIP_TAG_RE = new RegExp(
	`<\\s*\\/?\\s*${TAG_PREFIX}\\s*\\/?\\s*(?:invoke|function|parameter|calls?|tool_call)\\b[^>]*>`,
	"gi",
);
// `string="true"` marks values the model meant as a raw string rather than JSON.
const PARAMETER_STRING_ATTR_RE = /\bstring\s*=\s*(?:"true"|'true'|true)/i;

function attrString(match: RegExpMatchArray | null): string {
	return (match?.[1] ?? match?.[2] ?? match?.[3] ?? "").trim();
}

function decodeAttributeValue(value: string): string {
	return value
		.replace(/&quot;/g, '"')
		.replace(/&#39;|&apos;/g, "'")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&amp;/g, "&");
}

// Coerce a parameter body into the JSON value pi's tool arguments need. The
// model may send real JSON, a JSON string literal, or bare text.
function coerceParameterValue(raw: string, isStringAttr: boolean): unknown {
	const value = raw.trim();
	if (!isStringAttr && value) {
		try {
			return JSON.parse(value);
		} catch {
			// Not JSON; fall through and treat it as text.
		}
	}
	return decodeAttributeValue(value);
}

// Collect the `parameter` children of one invoke/function body.
function collectArguments(body: string): Record<string, unknown> {
	const args: Record<string, unknown> = {};
	PARAMETER_RE.lastIndex = 0;
	let parameter: RegExpExecArray | null;
	while ((parameter = PARAMETER_RE.exec(body)) !== null) {
		const [, paramAttrs = "", raw = ""] = parameter;
		const paramName = attrString(NAME_ATTR_RE.exec(paramAttrs));
		if (!paramName) continue;
		args[paramName] = coerceParameterValue(
			raw,
			PARAMETER_STRING_ATTR_RE.test(paramAttrs),
		);
	}
	return args;
}

// Does this text look like a model emitted a tool call as text?
function looksLikeToolMarkup(text: string, specs: ToolSpec[]): boolean {
	return (
		new RegExp(`<\\s*${TAG_PREFIX}\\s*invoke\\b`, "i").test(text) ||
		/<\s*function\b/i.test(text) ||
		/<\s*tool_call\s*>/i.test(text) ||
		specs.some((spec) => toolBlockRe(spec.name).test(text))
	);
}

// ---- Schema-aware tool blocks -------------------------------------------
// A third family names the *tool* as the element and each argument as a child
// element, with no invocation wrapper:
//
//   [LT]bash[GT]
//   [LT]command[GT]pwd[/command]
//   [LT]description[GT]Print current working directory[/description]
//   [/bash]
//
// Nothing in the text says whether `bash` is a tool or ordinary prose, so the
// live tool schemas decide: the element name must be a tool pi offered, and
// every child must be one of that tool's declared properties. Unknown children
// are dropped rather than sent as bogus arguments. `recoverySpecs` holds the
// schemas from the request that produced this stream.

type ToolSpec = {
	name: string;
	properties: Map<string, { type?: string; items?: { type?: string } }>;
	required: string[];
};

let recoverySpecs: ToolSpec[] = [];

function setRecoverySpecs(specs: ToolSpec[]): void {
	recoverySpecs = specs;
}

function escapeRegExp(literal: string): string {
	return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ---- Tool-name resolution ------------------------------------------------
// The model names tools the way it has seen them named elsewhere: `Bash`,
// `read_file`, `edit_file`, `glob`, `explore`. pi dispatches the name
// verbatim, so an unresolved name surfaces as "Tool bash not found" and the
// call is wasted — worse than not recovering it, because the model then retries
// the same wrong name. Recovered names (and the gateway's own `toolName`) are
// therefore resolved against the tools this request actually declared:
//
//   1. exact      `bash`      -> `bash`
//   2. case       `Bash`      -> `bash`
//   3. alias      `read_file` -> `read`   (only if `read` is declared)
//   4. arg shape  `glob` {pattern} -> the one declared tool that needs `pattern`
//
// Nothing resolves when the request declared no tools (print mode without
// `--tools`) or when no candidate is declared; the name is then left alone so
// the failure stays visible instead of silently swallowing the call.

const TOOL_NAME_ALIASES: Record<string, string[]> = {
	bash: ["bash_tool", "run_bash", "execute_bash", "shell", "terminal"],
	read: ["read_file", "view_file", "open_file", "cat"],
	edit: ["edit_file", "apply_edit", "str_replace", "patch_file"],
	write: ["write_file", "create_file", "save_file"],
	edit_file: ["edit_file"],
	grep: ["grep_search", "search_files", "ripgrep"],
	ffgrep: ["grep", "grep_search", "search", "rg"],
	find: ["glob", "find_files", "list_files"],
	fffind: ["glob", "find_files", "list_files"],
	ls: ["list_dir", "ls_dir", "list_directory"],
};

function resolveToolName(name: string, specs: ToolSpec[]): string | null {
	if (!name) return null;
	if (specs.length === 0) return null;

	const exact = specs.find((spec) => spec.name === name);
	if (exact) return exact.name;

	const lower = name.toLowerCase();
	const byCase = specs.find((spec) => spec.name.toLowerCase() === lower);
	if (byCase) return byCase.name;

	for (const [canonical, aliases] of Object.entries(TOOL_NAME_ALIASES)) {
		if (!aliases.includes(lower)) continue;
		const target =
			specs.find((spec) => spec.name.toLowerCase() === canonical) ??
			specs.find((spec) => spec.name.toLowerCase() === `${canonical}_tool`);
		if (target) return target.name;
	}

	return null;
}

// Last resort: match by the arguments the call carries. `edit_file` with
// `path` + `edits` can only be the declared tool that requires exactly those.
// Ambiguity stays unresolved rather than guessing.
function matchToolByArguments(
	args: Record<string, unknown>,
	specs: ToolSpec[],
): string | null {
	const keys = Object.keys(args);
	if (keys.length === 0) return null;
	const candidates = specs.filter(
		(spec) =>
			spec.required.length > 0 &&
			spec.required.every((key) => keys.includes(key)),
	);
	if (candidates.length !== 1) return null;
	return candidates[0].name;
}

function resolveRecoveredName(
	call: RecoveredToolCall,
	specs: ToolSpec[],
): string {
	return (
		resolveToolName(call.name, specs) ??
		matchToolByArguments(call.arguments, specs) ??
		call.name
	);
}

// ---- Argument normalization ----------------------------------------------
// A resolved tool still needs arguments pi's schema accepts. The model, primed
// by other harnesses' conventions, spells the same argument differently:
//
//   read_file { file_path }       -> read   { path }
//   edit_file { path, old_string, new_string } -> edit { path, edits: [{oldText,newText}] }
//
// Unknown keys are dropped: pi validates declared arguments, and a stray
// `description` on `bash` would otherwise reject the whole call. Values keep
// the type their declared schema wants.

const PROPERTY_ALIASES: Record<string, string[]> = {
	path: ["file_path", "filepath", "filePath", "filename", "file"],
	command: ["cmd", "shell_command", "script"],
	content: ["file_text", "fileText", "text", "contents"],
	pattern: ["query", "search", "regex", "glob_pattern"],
	limit: ["max_results", "maxResults", "count"],
	oldText: ["old_string", "oldString", "old_text", "search", "find"],
	newText: ["new_string", "newString", "new_text", "replace", "replacement"],
	edits: ["replacements", "changes", "patch"],
};

function canonicalProperty(key: string, spec: ToolSpec): string | null {
	if (spec.properties.has(key)) return key;
	// Aliases work in both directions: `file_path` finds `path`, and a declared
	// `file_path` is found from the canonical `path`.
	for (const [declared, aliases] of Object.entries(PROPERTY_ALIASES)) {
		if (!spec.properties.has(declared)) continue;
		const squashed = key.replace(/_/g, "").toLowerCase();
		if (
			aliases.some(
				(alias) =>
					alias === key || alias.replace(/_/g, "").toLowerCase() === squashed,
			)
		)
			return declared;
	}
	// A `filePath`/`file_path`-style variant of a declared camelCase property.
	const squashed = key.replace(/_/g, "").toLowerCase();
	for (const declared of spec.properties.keys()) {
		if (declared.replace(/_/g, "").toLowerCase() === squashed) return declared;
	}
	return null;
}

// pi's `edit` tool takes `edits` as an array of {oldText, newText}. Models that
// know a str_replace-style tool send the pair as flat arguments instead.
function synthesizeEdits(
	args: Record<string, unknown>,
	spec: ToolSpec,
): void {
	if (!spec.properties.has("edits") || args.edits !== undefined) return;
	const old =
		args.oldText ?? args.old_string ?? args.oldString ?? args.old_text;
	const replacement =
		args.newText ?? args.new_string ?? args.newString ?? args.new_text;
	if (typeof old !== "string" || typeof replacement !== "string") return;
	args.edits = [{ oldText: old, newText: replacement }];
}

function normalizeCallArguments(
	args: Record<string, unknown>,
	spec: ToolSpec | undefined,
): Record<string, unknown> {
	if (!spec) return args;
	synthesizeEdits(args, spec);

	const normalized: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(args)) {
		const declared = canonicalProperty(key, spec);
		if (!declared) continue;
		if (normalized[declared] !== undefined) continue;
		const schema = spec.properties.get(declared);
		normalized[declared] = coerceTypedValue(
			typeof value === "string" ? value : JSON.stringify(value),
			schema ?? {},
		);
	}
	return normalized;
}

function resolveCall(
	call: RecoveredToolCall,
	specs: ToolSpec[],
): RecoveredToolCall {
	const name = resolveRecoveredName(call, specs);
	const spec = specs.find((entry) => entry.name === name);
	return { name, arguments: normalizeCallArguments(call.arguments, spec) };
}

function toolBlockRe(name: string): RegExp {
	const TAG = escapeRegExp(name);
	return new RegExp(
		`<\\s*${TAG}\\s*>([\\s\\S]*?)<\\s*\\/\\s*${TAG}\\s*>`,
		"gi",
	);
}

function readSchema(schemaJson: unknown, name: string): ToolSpec | null {
	try {
		const schema =
			typeof schemaJson === "string" ? JSON.parse(schemaJson) : schemaJson;
		const props = (schema as { properties?: Record<string, unknown> })
			?.properties;
		if (!props || typeof props !== "object") return null;
		const required = (schema as { required?: unknown }).required;
		return {
			name,
			properties: new Map(
				Object.entries(props).map(([key, value]) => [
					key,
					(value ?? {}) as { type?: string; items?: { type?: string } },
				]),
			),
			required: Array.isArray(required)
				? required.filter((key): key is string => typeof key === "string")
				: [],
		};
	} catch {
		return null;
	}
}

// Build the schemas for the tools in the outgoing request so the text pass can
// tell a tool block from prose.
function recoverySpecsFrom(tools: Array<{ name: string; input_schema?: unknown }>): ToolSpec[] {
	const specs: ToolSpec[] = [];
	for (const tool of tools) {
		const spec = readSchema(tool.input_schema, tool.name);
		if (spec) specs.push(spec);
	}
	return specs;
}

// Read one property value into the JSON type its schema declares.
function coerceTypedValue(
	raw: string,
	schema: { type?: string; items?: { type?: string } },
): unknown {
	const value = raw.trim();
	const decoded = decodeAttributeValue(value);

	if (schema.type === "array") {
		if (!value) return [];
		try {
			const parsed = JSON.parse(value);
			if (Array.isArray(parsed)) return parsed;
		} catch {
			// Not JSON; fall through to the line/newline forms below.
		}
		return decoded
			.split("\n")
			.map((line) => line.trim())
			.filter(Boolean);
	}

	if (schema.type === "number" || schema.type === "integer") {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : decoded;
	}

	if (schema.type === "boolean") {
		if (value === "true") return true;
		if (value === "false") return false;
	}

	if (schema.type === "object") {
		try {
			return JSON.parse(value);
		} catch {
			return decoded;
		}
	}

	return decoded;
}

// Collect the child elements of a tool block, keeping only declared properties.
function collectTypedArguments(body: string, spec: ToolSpec): Record<string, unknown> {
	const args: Record<string, unknown> = {};
	for (const [name, schema] of spec.properties) {
		const TAG = escapeRegExp(name);
		const match = new RegExp(
			`<\\s*${TAG}(?:\\s[^>]*)?>([\\s\\S]*?)<\\s*\\/\\s*${TAG}\\s*>`,
			"i",
		).exec(body);
		// An empty child element still means the caller sent a value.
		if (!match) continue;
		args[name] = coerceTypedValue(match[1] ?? "", schema);
	}
	return args;
}

// Recover tool blocks that name the tool as the element. Only an element that
// matches a live tool name counts, and it needs at least one declared child.
//
// The model sometimes skips the child element and puts the value straight in
// the body instead:
//
//   [LT]bash[GT]pwd[/bash]
//
// That is only unambiguous when the tool takes exactly one required property
// and the body is not itself markup, so both conditions are required.
function recoverToolBlocks(
	text: string,
	specs: ToolSpec[],
): RecoveredToolCall[] {
	const calls: RecoveredToolCall[] = [];
	for (const spec of specs) {
		const BLOCK_RE = toolBlockRe(spec.name);
		BLOCK_RE.lastIndex = 0;
		let block: RegExpExecArray | null;
		while ((block = BLOCK_RE.exec(text)) !== null) {
			const body = block[1] ?? "";
			const args = collectTypedArguments(body, spec);
			if (Object.keys(args).length > 0) {
				calls.push({ name: spec.name, arguments: args });
				continue;
			}

			if (spec.required.length !== 1) continue;
			const sole = spec.required[0];
			const schema = spec.properties.get(sole);
			if (!schema) continue;
			// Child markup means the element form was intended but malformed.
			if (/<\s*\/?\s*[A-Za-z_]/.test(body)) continue;
			const value = body.trim();
			if (!value) continue;
			// A bare tag in mid-sentence is prose about the markup, not a call. The
			// model starts a real one on its own line.
			const before = text.slice(0, block.index);
			if (!/(^|\n)[ \t]*$/.test(before)) continue;
			calls.push({
				name: spec.name,
				arguments: { [sole]: coerceTypedValue(value, schema) },
			});
		}
	}
	return calls;
}

type RecoveredToolCall = { name: string; arguments: Record<string, unknown> };

// A tool block whose body is a bare `Name: value` line rather than a child
// element, as in:
//
//   [LT]tool_call[GT]
//   Bash: echo HELLO_REAL
//   [/tool_call]
//
// The name is matched case-insensitively against the live tools, so `Bash`
// resolves to pi's `bash` instead of being dropped as unknown.
const NAME_LINE_RE = /^\s*([A-Za-z_][\w.-]*)\s*:\s*([\s\S]*)$/;

function recoverNameLineBlocks(
	text: string,
	specs: ToolSpec[],
): RecoveredToolCall[] {
	const calls: RecoveredToolCall[] = [];
	const byLower = new Map(specs.map((spec) => [spec.name.toLowerCase(), spec]));
	const WRAPPER_RE = /<\s*tool_call\s*>([\s\S]*?)<\s*\/\s*tool_call\s*>/gi;
	let wrapper: RegExpExecArray | null;
	while ((wrapper = WRAPPER_RE.exec(text)) !== null) {
		const match = NAME_LINE_RE.exec(wrapper[1] ?? "");
		if (!match) continue;
		const spec = byLower.get(match[1].toLowerCase());
		if (!spec) continue;
		const value = (match[2] ?? "").trim();
		if (!value) continue;
		if (spec.required.length !== 1) continue;
		const sole = spec.required[0];
		const schema = spec.properties.get(sole);
		if (!schema) continue;
		calls.push({ name: spec.name, arguments: { [sole]: coerceTypedValue(value, schema) } });
	}
	return calls;
}

// Parse every tool call in a text block, in the DSML invoke shape, the
// function-envelope shape, and the schema-matched tool-element shape. Returns
// [] when the text has no usable markup, so callers can leave ordinary prose
// alone.
function recoverToolCalls(
	text: string,
	specs: ToolSpec[] = recoverySpecs,
): RecoveredToolCall[] {
	const calls: RecoveredToolCall[] = [];

	INVOKE_RE.lastIndex = 0;
	let invoke: RegExpExecArray | null;
	while ((invoke = INVOKE_RE.exec(text)) !== null) {
		const [, attrs = "", body = ""] = invoke;
		const name = attrString(NAME_ATTR_RE.exec(attrs));
		if (!name) continue;
		calls.push({ name, arguments: collectArguments(body) });
	}

	FUNCTION_RE.lastIndex = 0;
	let fn: RegExpExecArray | null;
	while ((fn = FUNCTION_RE.exec(text)) !== null) {
		const [, attrs = "", body = ""] = fn;
		const name = attrString(NAME_ATTR_RE.exec(attrs));
		if (!name) continue;
		calls.push({ name, arguments: collectArguments(body) });
	}

	calls.push(...recoverToolBlocks(text, specs));
	calls.push(...recoverNameLineBlocks(text, specs));

	// Resolve every recovered name against the declared tools before it can
	// reach pi, which dispatches names verbatim, and reshape the arguments to the
	// schema that name won.
	return calls.map((call) => resolveCall(call, specs));
}

// Strip the markup out of a text block so the leftover prose is not duplicated
// alongside the recovered tool calls.
function stripToolMarkup(text: string, specs: ToolSpec[] = recoverySpecs): string {
	let stripped = text.replace(INVOKE_RE, "").replace(FUNCTION_RE, "");
	stripped = stripped.replace(/<\s*tool_call\s*>[\s\S]*?<\s*\/\s*tool_call\s*>/gi, "");
	for (const spec of specs) stripped = stripped.replace(toolBlockRe(spec.name), "");
	return stripped
		.replace(STRIP_TAG_RE, "")
		.replace(/[ \t]+\n/g, "\n")
		.replace(/\n{3,}/g, "\n\n")
		.trim();
}

function convertMessages(messages: Message[]): CmdMessage[] {
	const out: CmdMessage[] = [];
	// Track tool call names by id so tool-result messages can fill `toolName`
	const toolNameById = new Map<string, string>();

	for (const msg of messages) {
		if (msg.role === "user") {
			const content: UserContent[] = [];
			if (typeof msg.content === "string") {
				content.push({ type: "text", text: msg.content });
			} else {
				for (const block of msg.content) {
					if (block.type === "text") content.push({ type: "text", text: block.text });
					else if (block.type === "image")
						content.push({
							type: "image",
							image: `data:${block.mimeType};base64,${block.data}`,
							mediaType: block.mimeType,
						});
				}
			}
			if (content.length > 0) out.push({ role: "user", content });
		} else if (msg.role === "assistant") {
			const content: AssistantContent[] = [];
			for (const block of msg.content) {
				if (block.type === "text") {
					if (block.text) content.push({ type: "text", text: block.text });
				} else if (block.type === "toolCall") {
					toolNameById.set(block.id, block.name);
					content.push({
						type: "tool-call",
						toolCallId: block.id,
						toolName: block.name,
						input: block.arguments ?? {},
					});
				}
				// Skip "thinking" — the upstream provider reconstructs its own reasoning
			}
			if (content.length > 0) out.push({ role: "assistant", content });
		} else if (msg.role === "toolResult") {
			const text = msg.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("");
			const block: ToolContent = {
				type: "tool-result",
				toolCallId: msg.toolCallId,
				toolName: toolNameById.get(msg.toolCallId) ?? msg.toolName ?? "unknown",
				output: { type: msg.isError ? "error-text" : "text", value: text },
			};
			// Merge consecutive tool-result messages into one role:"tool" message
			const last = out[out.length - 1];
			if (last && last.role === "tool") {
				last.content.push(block);
			} else {
				out.push({ role: "tool", content: [block] });
			}
		}
	}
	return out;
}

// ---- Static "config" sidecar --------------------------------------------
// The /alpha/generate schema is strict — all of these fields are required.
// We don't actually have a project context inside an extension, so we send
// neutral defaults. The gateway just stuffs them into the system prompt
// preamble; they don't affect routing.

function staticConfig() {
	return {
		workingDir: process.cwd(),
		date: new Date().toISOString().slice(0, 10),
		environment: "production",
		structure: [],
		isGitRepo: false,
		currentBranch: "",
		mainBranch: "",
		gitStatus: "",
		recentCommits: [],
	};
}

// ---- NDJSON line reader -------------------------------------------------

async function* ndjsonLines(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
	const reader = body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	let lineNumber = 0;
	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true });
		let nl: number;
		while ((nl = buffer.indexOf("\n")) >= 0) {
			const line = buffer.slice(0, nl).trim();
			buffer = buffer.slice(nl + 1);
			if (!line) continue;
			lineNumber += 1;
			yield parseJsonLine(line, lineNumber);
		}
	}
	buffer = buffer.trim();
	if (buffer) {
		lineNumber += 1;
		yield parseJsonLine(buffer, lineNumber);
	}
}

type GatewayEvent = Record<string, any>;

function toolEventId(event: GatewayEvent): string | undefined {
	return event.id ?? event.toolCallId;
}

function toolEventName(event: GatewayEvent): string {
	return event.toolName ?? event.name ?? "";
}

// ---- Transcript → request ------------------------------------------------
// The current tool set lives in the transcript's system messages. Return null
// when no system message declares any, so callers can fall back to the legacy
// top-level `context.tools` field instead of sending an empty tool list.
function toolsFromTranscript(context: Context): Tool[] | null {
	const counts = context.messages.some(
		(message) =>
			message.role === "system" &&
			(((message.toolsAdded?.length ?? 0) > 0) ||
				((message.toolsRemoved?.length ?? 0) > 0)),
	);
	if (!counts) return null;
	return getCurrentTools(context.messages);
}

// ---- Stream implementation ----------------------------------------------

function streamCommandCode(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const apiKey = options?.apiKey;
			if (!apiKey) throw new Error("No Command Code API key. Set COMMANDCODE_API_KEY=user_...");

			stream.push({ type: "start", partial: output });

			// pi hands providers a TranscriptContext: the prompt and the tool
			// declarations live in the transcript's system messages (`sections`,
			// `toolsAdded`), not in top-level `systemPrompt`/`tools` fields. Reading
			// those directly yields an empty tool list, so the model is offered no
			// tools at all and answers with tool-call markup in the text channel
			// instead of `tool-input-*` events. Replay the transcript to recover both.
			// The top-level fields are still honoured for older cores and for direct
			// `streamSimple()` calls from tests and other SDK users.
			const declaredTools =
				(toolsFromTranscript(context) ?? context.tools ?? []);
			const systemPrompt =
				getCurrentSystemPrompt(context.messages) ||
				context.systemPrompt ||
				"";

			const tools = declaredTools.map((t) => ({
				name: t.name,
				description: t.description,
				input_schema: t.parameters,
			}));
			setRecoverySpecs(recoverySpecsFrom(tools));

			const body = {
				config: staticConfig(),
				memory: "",
				taste: null,
				skills: null,
				permissionMode: "standard",
				params: {
					model: model.id,
					system: systemPrompt,
					messages: convertMessages(context.messages),
					tools,
					max_tokens: options?.maxTokens ?? model.maxTokens ?? 8192,
					...(options?.temperature !== undefined ? { temperature: options.temperature } : {}),
					stream: true,
				},
			};

			const response = await fetch(`${BASE_URL}${ENDPOINT}`, {
				method: "POST",
				headers: {
					Authorization: `Bearer ${apiKey}`,
					"Content-Type": "application/json",
					Accept: "application/x-ndjson",
					"x-cli-environment": "production",
					"x-command-code-version": COMMAND_CODE_VERSION,
					"x-session-id": options?.sessionId ?? crypto.randomUUID(),
					...(process.env.CMD_ZDR === "1" ? { "x-cmd-zdr": "1" } : {}),
				},
				body: JSON.stringify(body),
				signal: options?.signal,
			});

			if (!response.ok || !response.body) {
				let detail = "";
				try {
					detail = await response.text();
				} catch {}
				throw new Error(`Command Code ${response.status}: ${detail || response.statusText}`);
			}

			// id-keyed maps: gateway gives us "reasoning-0", "txt-0", "call_..." as ids
			const idToIndex = new Map<string, number>();
			const toolJsonByIndex = new Map<number, string>();
			const endedToolCalls = new Set<number>();
			let sawTerminalEvent = false;

			for await (const event of ndjsonLines(response.body)) {
				if (!event || typeof event !== "object") continue;
				const gatewayEvent = event as GatewayEvent;
				const type = gatewayEvent.type;
				if (!type) continue;

				switch (type) {
					case "reasoning-start": {
						output.content.push({ type: "thinking", thinking: "" });
						const idx = output.content.length - 1;
						idToIndex.set(gatewayEvent.id, idx);
						stream.push({ type: "thinking_start", contentIndex: idx, partial: output });
						break;
					}
					case "reasoning-delta": {
						const idx = idToIndex.get(gatewayEvent.id);
						if (idx === undefined) break;
						const block = output.content[idx];
						if (block.type !== "thinking") break;
						const delta = gatewayEvent.text ?? "";
						block.thinking += delta;
						stream.push({ type: "thinking_delta", contentIndex: idx, delta, partial: output });
						break;
					}
					case "reasoning-end": {
						const idx = idToIndex.get(gatewayEvent.id);
						if (idx === undefined) break;
						const block = output.content[idx];
						if (block.type !== "thinking") break;
						stream.push({
							type: "thinking_end",
							contentIndex: idx,
							content: block.thinking,
							partial: output,
						});
						break;
					}
					case "text-start": {
						output.content.push({ type: "text", text: "" });
						const idx = output.content.length - 1;
						idToIndex.set(gatewayEvent.id, idx);
						stream.push({ type: "text_start", contentIndex: idx, partial: output });
						break;
					}
					case "text-delta": {
						const idx = idToIndex.get(gatewayEvent.id);
						if (idx === undefined) break;
						const block = output.content[idx];
						if (block.type !== "text") break;
						const delta = gatewayEvent.text ?? "";
						block.text += delta;
						stream.push({ type: "text_delta", contentIndex: idx, delta, partial: output });
						break;
					}
					case "text-end": {
						const idx = idToIndex.get(gatewayEvent.id);
						if (idx === undefined) break;
						const block = output.content[idx];
						if (block.type !== "text") break;
						stream.push({
							type: "text_end",
							contentIndex: idx,
							content: block.text,
							partial: output,
						});
						break;
					}
					case "tool-input-start": {
						const id = toolEventId(gatewayEvent);
						if (!id) break;
						const name = resolveToolName(toolEventName(gatewayEvent), recoverySpecs) ?? toolEventName(gatewayEvent);
						output.content.push({
							type: "toolCall",
							id,
							name,
							arguments: {},
						});
						const idx = output.content.length - 1;
						idToIndex.set(id, idx);
						toolJsonByIndex.set(idx, "");
						stream.push({ type: "toolcall_start", contentIndex: idx, partial: output });
						break;
					}
					case "tool-input-delta": {
						const id = toolEventId(gatewayEvent);
						if (!id) break;
						const idx = idToIndex.get(id);
						if (idx === undefined) break;
						const block = output.content[idx];
						if (block.type !== "toolCall") break;
						const delta = gatewayEvent.delta ?? "";
						const acc = (toolJsonByIndex.get(idx) ?? "") + delta;
						toolJsonByIndex.set(idx, acc);
						try {
							block.arguments = JSON.parse(acc);
						} catch {
							// JSON still streaming
						}
						stream.push({ type: "toolcall_delta", contentIndex: idx, delta, partial: output });
						break;
					}
					case "tool-input-end":
					case "tool-call": {
						const id = toolEventId(gatewayEvent);
						if (!id) break;
						let idx = idToIndex.get(id);
						if (idx === undefined) {
							const rawName = toolEventName(gatewayEvent);
							output.content.push({
								type: "toolCall",
								id,
								name: resolveToolName(rawName, recoverySpecs) ?? rawName,
								arguments: {},
							});
							idx = output.content.length - 1;
							idToIndex.set(id, idx);
							stream.push({ type: "toolcall_start", contentIndex: idx, partial: output });
						}
						const block = output.content[idx];
						if (block.type !== "toolCall") break;
						// Some streams send full input on "tool-call"; prefer that if present
						const completeInput = gatewayEvent.input ?? gatewayEvent.args;
						if (completeInput && typeof completeInput === "object") {
							block.arguments = completeInput;
						} else {
							const acc = toolJsonByIndex.get(idx) ?? "";
							if (acc) {
								try {
									block.arguments = JSON.parse(acc);
								} catch {}
							}
						}
						if (endedToolCalls.has(idx)) break;
						endedToolCalls.add(idx);
						// A structured call can still carry another harness's argument spelling;
						// normalise it to the schema of the tool that was actually declared.
						block.arguments = normalizeCallArguments(
							block.arguments as Record<string, unknown>,
							recoverySpecs.find((spec) => spec.name === block.name),
						) as JsonObject;
						stream.push({
							type: "toolcall_end",
							contentIndex: idx,
							toolCall: {
								type: "toolCall",
								id: block.id,
								name: block.name,
								arguments: block.arguments,
							},
							partial: output,
						});
						break;
					}
					case "finish-step":
					case "finish": {
						sawTerminalEvent = true;
						const usage = gatewayEvent.usage ?? gatewayEvent.totalUsage;
						if (usage) {
							// The gateway reports inputTokens as the TOTAL input (cached + uncached),
							// matching the Vercel AI SDK convention. Pi's Usage shape expects
							// `input` and `cacheRead` to be disjoint — calculateCost multiplies
							// each separately, so leaving cached tokens inside `input` would
							// double-charge on paid models. Subtract to match the convention
							// used by the built-in Anthropic provider in pi-ai.
							const totalInputTokens = usage.inputTokens ?? usage.input_tokens ?? 0;
							const cacheReadTokens =
								usage.cachedInputTokens ??
								usage.inputTokenDetails?.cacheReadTokens ??
								usage.raw?.prompt_cache_hit_tokens ??
								0;
							output.usage.input = Math.max(0, totalInputTokens - cacheReadTokens);
							output.usage.output = usage.outputTokens ?? usage.output_tokens ?? 0;
							output.usage.cacheRead = cacheReadTokens;
							output.usage.cacheWrite = 0;
							output.usage.totalTokens =
								output.usage.input +
								output.usage.output +
								output.usage.cacheRead +
								output.usage.cacheWrite;
							calculateCost(model, output.usage);
						}
						const reason = gatewayEvent.finishReason ?? gatewayEvent.rawFinishReason;
						const sawToolCall = output.content.some((b) => b.type === "toolCall");
						if (reason === "length") output.stopReason = "length";
						else if (reason === "tool-calls" || reason === "tool_calls" || reason === "tool_use")
							output.stopReason = "toolUse";
						// Some OSS models report finishReason "stop" even when they emitted
						// tool calls; pi must still route those as tool use (mirrors the
						// cliproxy commandcode translator).
						else if (sawToolCall) output.stopReason = "toolUse";
						else output.stopReason = "stop";
						break;
					}
					case "error": {
						throw new Error(
							gatewayEvent.message ??
								(gatewayEvent.error === undefined
									? "Command Code stream error"
									: stringifyUnknown(gatewayEvent.error)),
						);
					}
				}
			}
			if (!sawTerminalEvent) {
				throw new Error("Command Code stream ended before a terminal event");
			}

			// A model can answer with tool-call markup in the text channel instead of
			// `tool-input-*` events. Recover those before reporting "stop", or pi sees
			// an ordinary answer and the agent loop stalls. Only runs when the gateway
			// sent no structured tool calls at all, so it cannot double up.
			if (!output.content.some((block) => block.type === "toolCall")) {
				for (let idx = 0; idx < output.content.length; idx++) {
					const block = output.content[idx];
					if (block.type !== "text") continue;
					if (!looksLikeToolMarkup(block.text, recoverySpecs)) continue;

					const recovered = recoverToolCalls(block.text, recoverySpecs);
					if (recovered.length === 0) continue;

					const residue = stripToolMarkup(block.text, recoverySpecs);
					if (residue) {
						block.text = residue;
					} else {
						// Nothing but markup: drop the empty text block.
						output.content.splice(idx, 1);
						idx--;
					}

					for (const call of recovered) {
						const id = `call_recovered_${idx}_${output.content.length}`;
						output.content.push({
							type: "toolCall",
							id,
							name: call.name,
							arguments: call.arguments as JsonObject,
						});
						stream.push({
							type: "toolcall_end",
							contentIndex: output.content.length - 1,
							toolCall: {
								type: "toolCall",
								id,
								name: call.name,
								arguments: call.arguments as JsonObject,
							},
							partial: output,
						});
					}
				}

				if (output.content.some((block) => block.type === "toolCall")) {
					output.stopReason = "toolUse";
				}
			}

			stream.push({
				type: "done",
				reason: output.stopReason as "stop" | "length" | "toolUse",
				message: output,
			});
			stream.end();
		} catch (error) {
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = stringifyUnknown(error);
			if (process.env.DEBUG) {
				console.error("[commandcode] stream error:", error);
			}
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
}

// ---- Extension entry point ----------------------------------------------

export default function (pi: ExtensionAPI) {
	pi.registerProvider("commandcode", {
		name: "Command Code",
		baseUrl: BASE_URL,
		apiKey: "COMMANDCODE_API_KEY",
		authHeader: true,
		api: "commandcode-generate",
		streamSimple: streamCommandCode,
		models: MODELS,
	});
}
