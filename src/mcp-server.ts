// MCP server for bridge. v0.2.0 exposes: spawn, handoff, pane_near, close,
// compose_brief, health.
//
// Not yet:
//   - dispatch (composite — landing in bridge-tools v0.4)
//   - personai_init (composite — landing soon)
//   - Consolidated MCP surface (crew/wire/wire-ipc/knowledge/knowledge-indexer
//     replace/wrap/pass-through curation)
//   - Hook discovery for bridge-X integration plugins (empty registry shipped)

import { execSync } from "child_process";

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import {
  Orchestrator,
  createBackend,
  detectTerminal,
  listThemes,
  loadTheme,
  resolveThemeDir,
} from "@agiterra/crew-tools";
import {
  importKeyPair,
  setPlan,
  registerOrRefresh,
  createAuthJwt,
  RpcClient,
  WireConnection,
  derivePublicKeyB64,
} from "@agiterra/wire-tools";

import {
  spawn,
  handoff,
  paneNear,
  close as closeComposite,
  composeBrief,
  health,
  type SpawnDeps,
} from "@agiterra/bridge-tools";
import type {
  SpawnOptions,
  BridgeHook,
} from "@agiterra/bridge-tools/types";

const PKG_VERSION = "0.2.0";

const mcp = new Server(
  { name: "bridge", version: PKG_VERSION },
  {
    capabilities: { tools: {} },
    instructions:
      "Bridge — the orchestrator's plugin. Single-call composite tools that collapse the orchestrator's N-step dances (spawn, handoff, close, etc.) into one call each. Use spawn to bring up a new agent with a finished task brief; bridge handles wire identity registration, crew launch, pane placement, and IPC kickoff in one shot. pane_near resolves 'right of babka'-style placement intent without spawning. compose_brief is the dry-run inspector. health is the cross-leg diagnostic. Bridge also re-exposes a curated subset of crew (agent/pane/tab/machine/theme) and wire (set_plan, heartbeat_*, register_agent) tools so an orchestrator can install bridge alone and get the full crew+wire surface — these proxy straight through to the same crew-tools/wire-tools code those plugins use.",
  },
);

// --- Curated crew + wire tool re-exposure (Bridge complete-suite) ---
//
// Bridge re-exposes a curated subset of crew and wire tools so an orchestrator
// can install ONLY bridge and still get crew + wire functionality. Each proxied
// tool's inputSchema is copied verbatim from its source plugin
// (crew-tools/src/mcp-server.ts, wire-tools/src/mcp-server.ts); crew handlers
// call deps.orchestrator.<method>(...) exactly as crew's CallTool switch does,
// and wire handlers call the same @agiterra/wire-tools functions the wire
// plugin uses, signed with the orchestrator's identity
// (deps.parent_agent_id / deps.parent_signing_key / deps.wire_url).
//
// EXCLUDED from the crew subset (overlap bridge composites): agent_launch,
// agent_close. (agent_register is INCLUDED — it's an orchestrator's SELF-
// registration, which no bridge composite covers; personas need it to seed
// their own crews.db row on boot. Added 2.0.1.) Wire-ipc/github/knowledge/*
// stay separate plugins.

interface ProxyTool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  handler: (args: Record<string, unknown>, deps: SpawnDeps) => Promise<unknown>;
}

/**
 * Resolve the caller's terminal session ID — replicates crew's callerSession()
 * helper. Used by the pane/tab/url proxy handlers that need the orchestrator's
 * own pane (tab_register/pane_register require it; pane_create/url_open/pane_close
 * use it as an optional anchor/guard). Bridge runs as the orchestrator's own MCP
 * server, so the same TTY/env resolution that works in crew works here.
 */
async function callerSession(deps: SpawnDeps): Promise<string | undefined> {
  const terminal = deps.orchestrator.terminal;
  if (terminal.name === "cmux" && process.env.CMUX_SURFACE_ID) {
    return process.env.CMUX_SURFACE_ID;
  }
  try {
    const tty = execSync(`ps -o tty= -p ${process.ppid}`, { encoding: "utf-8" }).trim();
    if (tty && tty !== "??") {
      const id = await terminal.sessionIdForTty(tty);
      if (id) return id;
    }
  } catch (e) {
    console.error(`[bridge] TTY lookup failed for ppid ${process.ppid}:`, e);
  }
  if (process.env.CMUX_SURFACE_ID) return process.env.CMUX_SURFACE_ID;
  const raw = process.env.ITERM_SESSION_ID;
  if (raw) return raw.split(":")[1];
  return undefined;
}

function titleCase(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// Crew tools — proxied to deps.orchestrator methods. inputSchemas copied
// verbatim from crew-tools/src/mcp-server.ts; arg→method mapping matches its
// CallTool switch exactly.
const CREW_PROXY_TOOLS: ProxyTool[] = [
  {
    name: "agent_send",
    description: "Send keystrokes to an agent's screen session (attached or headless). Returns { sent, landed, screen }: `landed` is true/false when the text is verifiable (printable, no trailing \\r/\\n) — confirming it actually appeared in the input — or null for control/submit keys (\\r, Esc, arrows) which leave no stable visible text. `screen` is the post-send hardcopy so you can confirm delivery rather than trust a blind sent:true. NOTE: AskUserQuestion menus ignore free text + digits as a SELECT — use arrows+Enter, or Esc to cancel then send the answer as a fresh prompt.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID" },
        text: { type: "string", description: "Text to send (use \\r for enter in screen sessions)" },
        session: { type: "string", description: "Screen session name — disambiguates when multiple sessions share an agent ID" },
      },
      required: ["id", "text"],
    },
    handler: async (a, deps) => {
      const sendRes = await deps.orchestrator.sendToAgent(
        a.id as string,
        a.text as string,
        (a.cc_session_id as string | undefined) ?? (a.session as string | undefined),
      );
      return { sent: true, landed: sendRes.landed, screen: sendRes.screen };
    },
  },
  {
    name: "agent_read",
    description: "Read an agent's current screen output. Works whether the agent is attached to a pane or running headless.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID" },
        cc_session_id: { type: "string", description: "Claude Code session ID — disambiguates when multiple instances share an agent ID" },
      },
      required: ["id"],
    },
    handler: async (a, deps) => ({
      output: await deps.orchestrator.readAgent(a.id as string, a.cc_session_id as string | undefined),
    }),
  },
  {
    name: "agent_list",
    description: "List agents with status, pane, and runtime. Optional filters narrow the result — useful when you only want attached agents (for routing) or headless ones (for cleanup). Dead agents are pruned by the boot-time reconciler, so the list only contains agents whose screen session was alive at last check.",
    inputSchema: {
      type: "object",
      properties: {
        attached: { type: "boolean", description: "If true, return only agents attached to a pane. If false, return only headless agents. Omit for all." },
        pane: { type: "string", description: "Return only agents attached to this exact pane." },
        with_ttl: { type: "boolean", description: "If true, return only agents that have a ttl_idle_minutes set (ephemeral spawns). If false, return only agents without a TTL (permanent)." },
      },
    },
    handler: async (a, deps) => {
      let agents = await deps.orchestrator.listAgents();
      if (typeof a.attached === "boolean") {
        agents = agents.filter((ag) => (ag.pane !== null) === a.attached);
      }
      if (typeof a.pane === "string") {
        agents = agents.filter((ag) => ag.pane === a.pane);
      }
      if (typeof a.with_ttl === "boolean") {
        agents = agents.filter((ag) => (ag.ttl_idle_minutes !== null) === a.with_ttl);
      }
      return agents;
    },
  },
  {
    name: "agent_interrupt",
    description: "Interrupt an agent. Returns screen output so you can assess the result.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID" },
        cc_session_id: { type: "string", description: "Claude Code session ID — disambiguates during handoff" },
        background: { type: "boolean", description: "If true, Ctrl-B Ctrl-B (background task). Default: Escape (cancel)." },
      },
      required: ["id"],
    },
    handler: async (a, deps) =>
      deps.orchestrator.interruptAgent(a.id as string, !!a.background, a.cc_session_id as string | undefined),
  },
  {
    name: "agent_resume",
    description:
      "Resume a stopped agent from its tombstone. Every agent_launch persists a spawn manifest (project_dir, env, channels, badge, ttl_idle_minutes, …). agent_stop copies that manifest into a tombstone row. Calling agent_resume({ id }) looks up the most recent tombstone for that id and reconstructs the spawn with one call — no re-supplying inputs.\n\nWire identity: AGENT_PRIVATE_KEY is stripped from the stored manifest, so if the agent uses Wire, first call register_agent({ id, force_rotate: true }) to mint a fresh keypair on Wire, then pass the returned private_key_b64 as env.AGENT_PRIVATE_KEY here. Wire's /agents/register is UPSERT keyed on id, so the same agent id continues to exist with a new pubkey.\n\nAny explicit field overrides the tombstone's value (including cc_session_id and project_dir — useful for resuming into a different worktree).",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID to resume (the id it was running under before agent_stop)." },
        cc_session_id: { type: "string", description: "Claude Code session ID — the JSONL filename stem. If omitted, falls back to the tombstone's cc_session_id (the session that was live when the agent was stopped)." },
        project_dir: { type: "string", description: "Working directory. Defaults to the tombstone manifest's project_dir." },
        env: { type: "object", additionalProperties: { type: "string" }, description: "Env overrides, merged on top of the tombstone's sanitized env. AGENT_PRIVATE_KEY is never in the tombstone — supply it here for Wire-using agents." },
        channels: { type: "array", items: { type: "string" }, description: "Dev-channel plugin list. Overrides the tombstone's recorded list; falls back to ['plugin:wire@agiterra']." },
        extra_flags: { type: "string", description: "Additional CLI flags appended after --resume. Defaults to tombstone's extra_flags." },
        attach_to_pane: { type: "string", description: "Optional pane to attach the resumed agent to once the screen is up." },
        display_name: { type: "string", description: "Display name. Defaults to tombstone's display_name." },
        badge: { type: "string", description: "Badge text. Defaults to tombstone's badge." },
        runtime: { type: "string", description: "Runtime. Defaults to tombstone's runtime. Only 'claude-code' is supported today." },
      },
      required: ["id"],
    },
    handler: async (a, deps) =>
      deps.orchestrator.resumeAgent({
        id: a.id as string,
        ccSessionId: a.cc_session_id as string | undefined,
        projectDir: a.project_dir as string | undefined,
        env: a.env as Record<string, string> | undefined,
        channels: a.channels as string[] | undefined,
        extraFlags: a.extra_flags as string | undefined,
        attachToPane: a.attach_to_pane as string | undefined,
        displayName: a.display_name as string | undefined,
        badge: a.badge as string | undefined,
        runtime: a.runtime as string | undefined,
      }),
  },
  {
    name: "agent_attach",
    description: "Attach an agent's screen session to a pane, making it visible. If another agent occupies the pane, it is detached first.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID" },
        pane: { type: "string", description: "Pane name" },
      },
      required: ["id", "pane"],
    },
    handler: async (a, deps) => {
      await deps.orchestrator.attachAgent(a.id as string, a.pane as string);
      return { attached: a.id, pane: a.pane };
    },
  },
  {
    name: "agent_detach",
    description: "Detach an agent from its pane. The agent keeps running headless in its screen session. The pane stays open with an empty shell.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID" },
      },
      required: ["id"],
    },
    handler: async (a, deps) => {
      await deps.orchestrator.detachAgent(a.id as string);
      return { detached: a.id };
    },
  },
  {
    name: "agent_move",
    description: "Move an agent to a different pane",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID" },
        pane: { type: "string", description: "Target pane name" },
      },
      required: ["id", "pane"],
    },
    handler: async (a, deps) => {
      await deps.orchestrator.moveAgent(a.id as string, a.pane as string);
      return { moved: a.id, pane: a.pane };
    },
  },
  {
    name: "agent_swap",
    description: "Swap two agents' panes",
    inputSchema: {
      type: "object",
      properties: {
        id_a: { type: "string", description: "First agent ID" },
        id_b: { type: "string", description: "Second agent ID" },
      },
      required: ["id_a", "id_b"],
    },
    handler: async (a, deps) => {
      await deps.orchestrator.swapAgents(a.id_a as string, a.id_b as string);
      return { swapped: [a.id_a, a.id_b] };
    },
  },
  {
    name: "agent_badge",
    description:
      "Set an agent's badge. Writes the text to the agent's DB row AND, if the agent is currently attached to a pane, pushes the text to that pane's iTerm2/cmux overlay immediately. This is the single call you should reach for when you want a badge to appear — no need to also call pane_badge. On subsequent agent_attach, the saved badge is re-rendered automatically. On agent_detach or agent_stop, the pane's badge is cleared. Badge color is determined by the pane's theme profile.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID" },
        text: { type: "string", description: "Badge text to display" },
      },
      required: ["id", "text"],
    },
    handler: async (a, deps) => {
      const outcome = await deps.orchestrator.setAgentBadge(a.id as string, a.text as string);
      return { badge_set: a.id, text: a.text, ...outcome };
    },
  },
  {
    name: "agent_register",
    description:
      "Register YOURSELF as a crew agent in crews.db. Call this ON BOOT (before agent_badge) if you run in a screen session. It reads your STY, verifies the screen is alive, seeds your registry row, and auto-links your pane if one exists with the same name. Personas launched via personai-launch (not crew spawn) get NO auto-row — self-register is the sanctioned path for an orchestrator to appear in the registry (e.g. after a crews.db migration/reset that dropped persona rows). This is your OWN row; to bring up another agent use spawn.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Your agent ID (your Wire agent name)" },
        name: { type: "string", description: "Display name" },
        runtime: { type: "string", description: "Runtime (default claude-code)" },
        cc_session_id: { type: "string", description: "Claude Code session ID — auto-detected from ~/.claude/sessions/ if omitted" },
      },
      required: ["id", "name"],
    },
    handler: async (a, deps) => {
      const agent = await deps.orchestrator.registerAgent({
        id: a.id as string,
        displayName: a.name as string,
        runtime: a.runtime as string | undefined,
        ccSessionId: a.cc_session_id as string | undefined,
        callerSessionId: await callerSession(deps),
      });
      return { registered: agent.id, screen_name: agent.screen_name, pane: agent.pane };
    },
  },
  {
    name: "agent_stop",
    description:
      "Hard-stop an agent (kills the screen session and all child processes). Use only for exceptional circumstances — runtime is unresponsive, hung agent, etc. For normal shutdown, prefer bridge's close/handoff, which let the runtime exit cleanly and fire its SessionEnd hooks (so e.g. ephemeral wire agents are removed from the dashboard immediately instead of greyed for an hour). The pane stays open — use pane_close separately if you want the pane gone too.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Agent ID" },
        cc_session_id: { type: "string", description: "Claude Code session ID — disambiguates when multiple instances share an agent ID (e.g. during handoff)" },
      },
      required: ["id"],
    },
    handler: async (a, deps) => {
      await deps.orchestrator.stopAgent(a.id as string, a.cc_session_id as string | undefined);
      return { stopped: a.id, cc_session_id: a.cc_session_id };
    },
  },
  {
    name: "pane_create",
    description: "Create a pane by splitting an existing terminal pane.",
    inputSchema: {
      type: "object",
      properties: {
        tab: { type: "string", description: "Tab name" },
        name: { type: "string", description: "Pane name (optional — auto-assigned from tab theme if omitted)" },
        position: { type: "string", description: "Split direction: below (default), right, left, above" },
        relative_to: { type: "string", description: "Pane name or session ID to split from (default: tab's session or caller's pane)" },
      },
      required: ["tab"],
    },
    handler: async (a, deps) => {
      // Only fall back to caller's session if creating in the caller's OWN tab.
      const targetTab = a.tab as string;
      let relTo = a.relative_to as string | undefined;
      if (!relTo) {
        const callerId = await callerSession(deps);
        if (callerId) {
          const callerPane = deps.orchestrator.store.listPanes().find((p) => p.iterm_id === callerId);
          if (callerPane && callerPane.tab === targetTab) relTo = callerId;
        }
      }
      return deps.orchestrator.createPane(
        targetTab,
        a.name as string | undefined,
        a.position as string | undefined,
        relTo,
      );
    },
  },
  {
    name: "pane_close",
    description: "Close a pane. Detaches any agent first. NEVER close a pane you are sitting in.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Pane name" },
      },
      required: ["name"],
    },
    handler: async (a, deps) => {
      await deps.orchestrator.closePane(a.name as string, await callerSession(deps));
      return { closed: a.name };
    },
  },
  {
    name: "pane_list",
    description: "List all panes, optionally filtered by tab",
    inputSchema: {
      type: "object",
      properties: {
        tab: { type: "string", description: "Optional tab filter" },
      },
    },
    handler: async (a, deps) => deps.orchestrator.listPanes(a.tab as string | undefined),
  },
  {
    name: "pane_badge",
    description:
      "Set a badge on a pane directly, bypassing any agent that may be attached. Use this for pane-purpose labels on panes that don't currently host an agent (e.g., an empty pane reserved for a future role, a tab-metadata slot). For agent-identity badges, prefer agent_badge — it writes the DB and auto-renders when the agent attaches, and it survives detach/reattach cycles. A badge set via pane_badge is transient — an agent attaching to the pane will overwrite it with its own badge.",
    inputSchema: {
      type: "object",
      properties: {
        pane: { type: "string", description: "Pane name" },
        text: { type: "string", description: "Badge text" },
      },
      required: ["pane", "text"],
    },
    handler: async (a, deps) => {
      await deps.orchestrator.setBadge(a.pane as string, a.text as string);
      return { badge_set: a.pane, text: a.text };
    },
  },
  {
    name: "pane_notify",
    description: "Flash a pane's tab and send a notification. On cmux: triggers the notification ring + desktop alert. On iTerm2: sets badge text.",
    inputSchema: {
      type: "object",
      properties: {
        pane: { type: "string", description: "Pane name" },
        title: { type: "string", description: "Notification title" },
        body: { type: "string", description: "Notification body (optional)" },
      },
      required: ["pane", "title"],
    },
    handler: async (a, deps) => {
      await deps.orchestrator.notifyPane(a.pane as string, a.title as string, a.body as string | undefined);
      return { notified: a.pane, title: a.title };
    },
  },
  {
    name: "pane_register",
    description: "Register your own terminal pane. Call this at session start so other agents can split relative to your pane.",
    inputSchema: {
      type: "object",
      properties: {
        tab: { type: "string", description: "Tab name (created if missing)" },
        name: { type: "string", description: "Pane name (optional — auto-assigned from tab theme if omitted)" },
      },
      required: ["tab"],
    },
    handler: async (a, deps) => {
      const sessionId = await callerSession(deps);
      if (!sessionId) throw new Error(`cannot detect terminal session — are you running in ${deps.orchestrator.terminal.name}?`);
      if (!deps.orchestrator.store.getTab(a.tab as string)) {
        await deps.orchestrator.createTab(a.tab as string);
      }
      return deps.orchestrator.registerPane(a.tab as string, a.name as string | undefined, sessionId);
    },
  },
  {
    name: "tab_create",
    description: "Create a named tab (a container for panes) by SPAWNING a new iTerm tab via AppleScript. To bind an existing iTerm tab, use tab_register instead. Optionally set a theme for auto-naming panes.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Tab name" },
        theme: { type: "string", description: "Pane naming theme: trees, rivers, stones, peaks, spices, cities. Panes created without a name get one from this pool." },
      },
      required: ["name"],
    },
    handler: async (a, deps) => deps.orchestrator.createTab(a.name as string, a.theme as string | undefined),
  },
  {
    name: "tab_destroy",
    description: "Destroy a tab and all its panes. Agents in those panes are detached (keep running headless). NEVER destroy a tab containing a pane you are sitting in.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Tab name" },
      },
      required: ["name"],
    },
    handler: async (a, deps) => {
      deps.orchestrator.deleteTab(a.name as string);
      return { destroyed: a.name };
    },
  },
  {
    name: "tab_list",
    description: "List all tabs",
    inputSchema: { type: "object", properties: {} },
    handler: async (_a, deps) => deps.orchestrator.listTabs(),
  },
  {
    name: "tab_register",
    description: "Register an EXISTING terminal tab (the one you're already running in, or one the operator opened manually) without spawning a new iTerm tab via AppleScript. Use this when an agent boots in a manually-opened tab and wants crew to track it. Defaults iterm_session_id to the caller's current session.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Tab name" },
        theme: { type: "string", description: "Pane naming theme: trees, rivers, stones, peaks, spices, cities" },
        iterm_session_id: { type: "string", description: "Terminal session uuid (default: caller's current session)" },
      },
      required: ["name"],
    },
    handler: async (a, deps) => {
      const sessionId = (a.iterm_session_id as string | undefined) ?? (await callerSession(deps));
      if (!sessionId) throw new Error(`cannot detect terminal session — pass iterm_session_id explicitly or run from inside ${deps.orchestrator.terminal.name}`);
      return deps.orchestrator.registerTab(a.name as string, sessionId, a.theme as string | undefined);
    },
  },
  {
    name: "machine_list",
    description: "List all machines registered in the local crew DB. The local machine is auto-registered on first boot and appears as a row with ssh_host='localhost'.",
    inputSchema: { type: "object", properties: {} },
    handler: async (_a, deps) => deps.orchestrator.listMachines(),
  },
  {
    name: "machine_probe",
    description: "Re-probe a registered machine for reachability. Updates last_seen + crew_version on success; returns the probe result either way.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Machine name to probe." },
      },
      required: ["name"],
    },
    handler: async (a, deps) => deps.orchestrator.probeMachine(a.name as string),
  },
  {
    name: "machine_register",
    description:
      "Register a machine in the local crew DB so cross-machine tools (crew-fleet, fleet_move) can reach it. Probes SSH with BatchMode=yes + ConnectTimeout=5 and reads the remote crew plugin version. Each crew DB is local-truth for 'machines I know how to reach' — no central registry. Pass `reciprocal: true` to also register the local machine on the remote side.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "User-friendly alias (e.g. 'home-mini')." },
        ssh_host: { type: "string", description: "SSH destination (e.g. 'tim@mac-mini.local')." },
        ssh_port: { type: "number", description: "Optional SSH port. Default: 22." },
        broker_url: { type: "string", description: "Externally-reachable Wire broker URL for agents spawned ON this machine (e.g. 'https://patisserie.ngrok.io'). Required for cross-machine spawn (bridge `spawn` with `machine`): the sponsoring parent registers the new ephemeral against this url so the remote broker knows its key, and the agent receives it as WIRE_EXTERNAL_URL. Omit for machines with no distinct broker." },
        notes: { type: "string", description: "Free-form notes." },
        skip_probe: { type: "boolean", description: "Skip the SSH reachability check at register time (default false)." },
        reciprocal: { type: "boolean", description: "If true, after registering the destination, SSH to it and register the LOCAL machine in its DB. Best-effort — SSH failure during reciprocal step does not undo the local row." },
        local_address: { type: "string", description: "Override the SSH-reachable address of the local machine for the reciprocal call. Defaults to '${USER}@${hostname}.local'." },
      },
      required: ["name", "ssh_host"],
    },
    handler: async (a, deps) =>
      deps.orchestrator.registerMachine({
        name: a.name as string,
        sshHost: a.ssh_host as string,
        sshPort: a.ssh_port as number | undefined,
        brokerUrl: a.broker_url as string | undefined,
        notes: a.notes as string | undefined,
        skipProbe: Boolean(a.skip_probe),
        reciprocal: Boolean(a.reciprocal),
        localAddress: a.local_address as string | undefined,
      }),
  },
  {
    name: "machine_remove",
    description: "Remove a registered machine. Refuses to delete the local machine — its row is required for agents.machine_name joins.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Machine name to remove." },
      },
      required: ["name"],
    },
    handler: async (a, deps) => {
      deps.orchestrator.removeMachine(a.name as string);
      return { removed: a.name };
    },
  },
  {
    name: "theme_list",
    description: "List installed themes with pool coverage",
    inputSchema: { type: "object", properties: {} },
    handler: async () => {
      const themes = listThemes();
      return themes.map((n) => {
        const config = loadTheme(n);
        const dir = resolveThemeDir(n);
        const imageCount = config ? Object.keys(config.background.images).length : 0;
        const poolSize = config?.pool.length ?? 0;
        return {
          name: n, dir, pool: poolSize, images: imageCount,
          coverage: poolSize > 0 ? `${imageCount}/${poolSize}` : "no pool",
          blend: config?.background.blend, mode: config?.background.mode,
        };
      });
    },
  },
  {
    name: "theme_update",
    description: "Update a theme's blend, mode, or images, then rebuild all live panes using that theme.",
    inputSchema: {
      type: "object",
      properties: {
        theme: { type: "string", description: "Theme name" },
        blend: { type: "number", description: "Background blend/opacity (0-1)" },
        mode: { type: "number", description: "Background image mode (0=tile, 1=stretch, 2=scale-to-fill)" },
        images: { type: "object", description: "Map of pane name → image filename to update", additionalProperties: { type: "string" } },
      },
      required: ["theme"],
    },
    handler: async (a, deps) => {
      const updates: { blend?: number; mode?: number; images?: Record<string, string> } = {};
      if (a.blend !== undefined) updates.blend = a.blend as number;
      if (a.mode !== undefined) updates.mode = a.mode as number;
      if (a.images) updates.images = a.images as Record<string, string>;
      return deps.orchestrator.updateThemeAndRebuild(a.theme as string, updates);
    },
  },
  {
    name: "tombstone_list",
    description: "List tombstones (stopped-agent records) with their manifests. Useful for discovering what's resumable. Without `id`, returns the most recent tombstones across all agents; with `id`, returns all tombstones for that agent, most recent first.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Filter by agent id." },
        limit: { type: "number", description: "Max rows (default 50)." },
      },
    },
    handler: async (a, deps) =>
      deps.orchestrator.store.listTombstones(a.id as string | undefined, a.limit as number | undefined),
  },
  {
    name: "reconcile",
    description: "Sync DB state with running screen sessions.",
    inputSchema: { type: "object", properties: {} },
    handler: async (_a, deps) => ({ report: await deps.orchestrator.reconcile() }),
  },
  {
    name: "url_open",
    description: "Open a URL in a new pane",
    inputSchema: {
      type: "object",
      properties: {
        tab: { type: "string", description: "Tab name (must exist)" },
        pane: { type: "string", description: "Pane name (auto-generated if omitted)" },
        url: { type: "string", description: "URL to open" },
        position: { type: "string", description: "Split direction: below (default), right" },
        relative_to: { type: "string", description: "Pane name to split from" },
      },
      required: ["tab", "url"],
    },
    handler: async (a, deps) =>
      deps.orchestrator.openUrl({
        tab: a.tab as string,
        pane: a.pane as string | undefined,
        url: a.url as string,
        position: a.position as string | undefined,
        relativeTo: (a.relative_to as string) ?? (await callerSession(deps)),
      }),
  },
];

// Wire tools — proxied to @agiterra/wire-tools functions, signed with the
// orchestrator's identity (deps.parent_agent_id / deps.parent_signing_key /
// deps.wire_url). inputSchemas + REST shapes copied verbatim from
// wire-tools/src/mcp-server.ts. register_agent reuses registerOrRefresh — the
// same function bridge's own spawn() uses to sponsor-register agents.
const WIRE_PROXY_TOOLS: ProxyTool[] = [
  {
    name: "set_plan",
    description: "Update this agent's plan on the Wire dashboard",
    inputSchema: {
      type: "object",
      properties: {
        plan: { type: "string", description: "Plan text (shown on the Wire dashboard)" },
      },
      required: ["plan"],
    },
    handler: async (a, deps) => {
      await setPlan(deps.wire_url, deps.parent_agent_id, a.plan as string, deps.parent_signing_key);
      return "plan updated";
    },
  },
  {
    name: "heartbeat_create",
    description:
      "Create a scheduled heartbeat — a recurring prompt sent to an agent via Wire. Use this to wake yourself or an ephemeral agent up periodically to check on things.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string", description: "Agent to receive the heartbeat prompt. Defaults to self." },
        cron: { type: "string", description: "Cron expression (e.g. '*/5 * * * *' for every 5 minutes)" },
        prompt: { type: "string", description: "The prompt text sent to the agent on each tick" },
      },
      required: ["cron", "prompt"],
    },
    handler: async (a, deps) => {
      const agentId = (a.agent_id as string | undefined) ?? deps.parent_agent_id;
      const body = JSON.stringify({
        agent_id: agentId,
        cron: a.cron as string,
        prompt: a.prompt as string,
        created_by: deps.parent_agent_id,
      });
      const token = await createAuthJwt(deps.parent_signing_key, deps.parent_agent_id, body);
      const res = await fetch(`${deps.wire_url}/heartbeats`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body,
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      return res.json();
    },
  },
  {
    name: "heartbeat_list",
    description: "List all scheduled heartbeats, optionally filtered by agent.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string", description: "Filter by agent ID. Omit to list all." },
      },
    },
    handler: async (a, deps) => {
      const url = a.agent_id
        ? `${deps.wire_url}/heartbeats?agent_id=${a.agent_id as string}`
        : `${deps.wire_url}/heartbeats`;
      const token = await createAuthJwt(deps.parent_signing_key, deps.parent_agent_id, "");
      const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      return res.json();
    },
  },
  {
    name: "heartbeat_delete",
    description: "Delete a scheduled heartbeat by ID.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Heartbeat ID (from heartbeat_create or heartbeat_list)" },
      },
      required: ["id"],
    },
    handler: async (a, deps) => {
      const token = await createAuthJwt(deps.parent_signing_key, deps.parent_agent_id, "");
      const res = await fetch(`${deps.wire_url}/heartbeats/${a.id as string}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      return { deleted: a.id };
    },
  },
  {
    name: "register_agent",
    description:
      "Sponsor-register a Wire agent (signed by the orchestrator). Three modes (the tool picks based on args):\n\n" +
      "  (1) fresh — no pubkey supplied and id is unknown to Wire. Generates a fresh Ed25519 keypair, registers it as `id`, and returns `private_key_b64` for the caller to pass to crew agent_launch (or bridge spawn) as `env.AGENT_PRIVATE_KEY`.\n\n" +
      "  (2) refresh-existing — no pubkey supplied but Wire already has a row at this id. Reuses the existing pubkey so the live agent process (which still holds the matching private key) keeps working. NO private_key_b64 is returned in this mode.\n\n" +
      "  (3) byo — caller supplies `pubkey` (base64 raw Ed25519, 32 bytes). Skips keypair generation, registers the supplied pubkey. NO private_key_b64 is returned.\n\n" +
      "If an agent with this id already exists with a DIFFERENT pubkey, the call fails with HTTP 409 agent_exists_pubkey_mismatch unless you pass `force_rotate: true` — which permanently locks out any process still holding the old key.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "New agent's ID (the name it will register under and use as `env.AGENT_ID`)." },
        display_name: { type: "string", description: "Optional display name. Defaults to TitleCase(id)." },
        pubkey: { type: "string", description: "Optional. Base64 raw Ed25519 public key (32 bytes). When supplied, the tool skips keypair generation and registers this pubkey on Wire as `id`. Returns no private_key_b64." },
        force_rotate: { type: "boolean", description: "Default false. When true, mints a fresh keypair regardless of any existing row — permanently locking out any process still holding the previous private key. Use only when you've confirmed no live process holds the old key." },
      },
      required: ["id"],
      additionalProperties: false,
    },
    handler: async (a, deps) => {
      const id = a.id;
      const displayName = a.display_name;
      const forceRotate = a.force_rotate;
      const providedPubkey = a.pubkey;
      if (typeof id !== "string" || id.length === 0) {
        throw new Error(`register_agent: 'id' is required (string). Got: ${JSON.stringify(id)}.`);
      }
      if (displayName !== undefined && typeof displayName !== "string") {
        throw new Error(`register_agent: 'display_name' must be a string if provided. Got: ${JSON.stringify(displayName)}.`);
      }
      if (forceRotate !== undefined && typeof forceRotate !== "boolean") {
        throw new Error(`register_agent: 'force_rotate' must be a boolean if provided. Got: ${JSON.stringify(forceRotate)}.`);
      }
      if (providedPubkey !== undefined && typeof providedPubkey !== "string") {
        throw new Error(`register_agent: 'pubkey' must be a base64 string if provided. Got: ${JSON.stringify(providedPubkey)}.`);
      }
      const resolvedName = (displayName as string | undefined) ?? titleCase(id);
      const result = await registerOrRefresh(
        deps.wire_url,
        deps.parent_agent_id,
        deps.parent_signing_key,
        id,
        resolvedName,
        {
          pubkey: providedPubkey as string | undefined,
          force_rotate: forceRotate as boolean | undefined,
        },
      );
      const response: Record<string, string> = {
        agent_id: result.agentId,
        display_name: result.displayName,
        pubkey: result.pubkey,
        mode: result.mode,
      };
      if (result.privateKey) response.private_key_b64 = result.privateKey;
      return response;
    },
  },
];

// Combined proxy dispatch table — name → ProxyTool.
const PROXY_TOOLS: ReadonlyMap<string, ProxyTool> = new Map(
  [...CREW_PROXY_TOOLS, ...WIRE_PROXY_TOOLS].map((t) => [t.name, t]),
);

mcp.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "spawn",
      description:
        "Spawn a new agent end-to-end. Collapses the dance (wire register → env-map → crew.agent_spawn RPC → pane attach → IPC kickoff) into one call. The spawn is executed by THIS machine's crew-service (crew.agent_spawn), which owns the OS account (CREW_SVC_SPAWN_UID, e.g. _ephemeral) and creates the screen — the bridge never picks a uid. `roles` are opaque tags forwarded as AGENT_ROLES. `task` is the finished brief. `placement.near + direction` puts the new pane next to a known agent/pane, but ONLY applies when the service spawns under the bridge's own account; a sudo'd spawn (a spawn uid is configured) is headless and placement is ignored (attach later via the dashboard button / crew remote attach). Add `detached: true` to skip placement. `env` overrides per-spawn vars. `badge` (optional) is multi-line text shown in the pane's top-right when attached — typical format: 'Name — Role\\nTicket #ID'. `project_dir` is the spawn cwd — the agent loads its plugins from that dir's installed_plugins.json entries, so point it at a dir that has them (e.g. a project root, not a worktree subpath). `branch` (optional) is forwarded as the AGENT_BRANCH env hint; bridge does NOT create worktrees or manage layout — the agent makes its own worktree if it wants one. Spawning into a git-worktree subpath (`.../worktrees/<branch>`) is REJECTED with a clear error — the agent would load no plugins from there and launch IPC-blind; spawn at the repo root and pass `branch`. `force_rotate` (optional): if `agent_id` was previously reaped, pass `true` to mint a fresh Wire identity (otherwise registration 409s on the stale pubkey); only use when no live process holds the old key. Cross-MACHINE spawns are NOT supported here — spawns land on this machine's crew-service; to spawn elsewhere, ask an orchestrator on that machine. Returns {agent_id, wire_identity, applied_capabilities, brief_sent}.",
      inputSchema: {
        type: "object",
        properties: {
          agent_id: { type: "string" },
          display_name: { type: "string" },
          roles: { type: "array", items: { type: "string" } },
          task: { type: "string" },
          capabilities: { type: "array", items: { type: "string" } },
          placement: { type: "object" },
          env: { type: "object" },
          runtime: { type: "string" },
          project_dir: { type: "string" },
          badge: { type: "string" },
          branch: { type: "string" },
          force_rotate: { type: "boolean" },
        },
        required: ["agent_id", "roles", "task"],
      },
    },
    {
      name: "handoff",
      description:
        "Coordinated graceful exit of a worker agent. Publishes a bridge.handoff ack on Wire (so monitors know it's a graceful exit, not a reap), then calls crew.closeAgent (sends /exit, runs SessionEnd hooks, waits for clean shutdown), then optionally closes the pane. The closing agent is responsible for running /knowledge:save itself BEFORE signaling ready-to-close; bridge does not invoke that.",
      inputSchema: {
        type: "object",
        properties: {
          agent_id: { type: "string" },
          close_pane: { type: "string" },
          timeout_ms: { type: "number" },
        },
        required: ["agent_id"],
      },
    },
    {
      name: "pane_near",
      description:
        "Resolve 'near X, direction Y' placement intent into a concrete pane spec. Walks the crew tree to find the anchor (by pane name OR agent name → attached pane), then returns {tab, anchor_pane, direction, split_direction, via_agent}. Useful when the orchestrator wants to plan placement before spawning.",
      inputSchema: {
        type: "object",
        properties: {
          near: { type: "string" },
          direction: { type: "string", enum: ["right", "below", "left", "above"] },
        },
        required: ["near", "direction"],
      },
    },
    {
      name: "close",
      description:
        "Collapsed wrap-up dance: snapshot the agent's screen via crew.readAgent (so you can audit/journal it BEFORE the session is gone), then crew.closeAgent (graceful /exit), then optionally pane_close. No precondition enforcement — bridge collapses the mechanical dance; the orchestrator owns whatever discipline checks (Linear status, audit checklist) upstream.",
      inputSchema: {
        type: "object",
        properties: {
          agent_id: { type: "string" },
          close_pane: { type: "string" },
          skip_snapshot: { type: "boolean" },
          timeout_ms: { type: "number" },
        },
        required: ["agent_id"],
      },
    },
    {
      name: "compose_brief",
      description:
        "Dry-run preview of what spawn WOULD do without spawning. Returns assembled env (with AGENT_PRIVATE_KEY placeholder), resolved placement, hooks that would run, capabilities with no registered hook, and notes for orchestrator review. Hooks are NOT invoked (avoids side effects in dry-run). Takes the same arguments as spawn.",
      inputSchema: {
        type: "object",
        properties: {
          agent_id: { type: "string" },
          display_name: { type: "string" },
          roles: { type: "array", items: { type: "string" } },
          task: { type: "string" },
          capabilities: { type: "array", items: { type: "string" } },
          placement: { type: "object" },
          env: { type: "object" },
          runtime: { type: "string" },
          project_dir: { type: "string" },
          badge: { type: "string" },
          branch: { type: "string" },
          force_rotate: { type: "boolean" },
        },
        required: ["agent_id", "roles", "task"],
      },
    },
    {
      name: "health",
      description:
        "Cross-leg diagnostic. Pings the wire server, counts crew agents/panes/tabs, checks the knowledge vault path and journal.db presence. Read-only. Useful at session start or before a complex orchestration push.",
      inputSchema: {
        type: "object",
        properties: {
          vault_path: { type: "string" },
        },
      },
    },
    // Curated crew + wire tools re-exposed by bridge (see PROXY_TOOLS above).
    ...[...CREW_PROXY_TOOLS, ...WIRE_PROXY_TOOLS].map((t) => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    })),
  ],
}));

mcp.setRequestHandler(CallToolRequestSchema, async (req) => {
  if (!deps) {
    return {
      content: [{ type: "text" as const, text: "bridge: still initializing (or missing AGENT_ID / AGENT_PRIVATE_KEY / WIRE_URL env — check stderr log); retry in a few seconds" }],
      isError: true,
    };
  }

  const args = (req.params.arguments ?? {}) as Record<string, unknown>;

  try {
    let result: unknown;
    switch (req.params.name) {
      case "spawn":
        result = await spawn(args as unknown as SpawnOptions, deps, EMPTY_REGISTRY);
        break;
      case "handoff":
        result = await handoff(args as any, deps);
        break;
      case "pane_near":
        result = paneNear(args as any, { orchestrator: deps.orchestrator });
        break;
      case "close":
        result = await closeComposite(args as any, { orchestrator: deps.orchestrator });
        break;
      case "compose_brief":
        result = composeBrief(
          args as unknown as SpawnOptions,
          { orchestrator: deps.orchestrator, wire_url: deps.wire_url, parent_agent_id: deps.parent_agent_id },
          EMPTY_REGISTRY,
        );
        break;
      case "health":
        result = await health(args as any, { orchestrator: deps.orchestrator, wire_url: deps.wire_url });
        break;
      default: {
        // Curated crew + wire proxy tools re-exposed by bridge.
        const proxy = PROXY_TOOLS.get(req.params.name);
        if (!proxy) throw new Error(`unknown tool: ${req.params.name}`);
        result = await proxy.handler(args, deps);
        break;
      }
    }
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }],
    };
  } catch (e: unknown) {
    const err = e as Error;
    return {
      content: [{ type: "text" as const, text: `${req.params.name} failed: ${err.message}\n${err.stack ?? ""}` }],
      isError: true,
    };
  }
});

let deps: SpawnDeps | undefined;
const EMPTY_REGISTRY: ReadonlyMap<string, BridgeHook> = new Map();

export async function startServer(): Promise<void> {
  // v2.0.3: connect the MCP transport BEFORE any backend/wire init. On headless
  // hosts the iTerm2 osascript enumeration inside createBackend() can stall past
  // the MCP client's connect timeout — the client then abandons the handshake and
  // the session gets ZERO bridge tools while the server still logs "[bridge] ready"
  // (observed on brioche's Fable boot, 2026-07-16). Tools are listed immediately;
  // calls return a clear "still initializing" error until initDeps() finishes.
  const transport = new StdioServerTransport();
  await mcp.connect(transport);
  initDeps().catch((e) => {
    console.error(`[bridge] init failed (tools will keep erroring until restart): ${(e as Error).stack ?? e}`);
  });
}

async function initDeps(): Promise<void> {
  const AGENT_ID = process.env.AGENT_ID;
  const rawKey = process.env.AGENT_PRIVATE_KEY;
  const WIRE_URL = process.env.WIRE_URL;
  // Optional — used by spawn to forward into the spawned agent's env so
  // plugins that advertise webhook URLs (github-tools/register_pr_webhook,
  // slack-tools/register_slack_app) get a publicly-reachable URL.
  const WIRE_EXTERNAL_URL = process.env.WIRE_EXTERNAL_URL;
  // Optional — the BARE host this orchestrator runs on (ssh-reachable as
  // `tim@<host>`). spawn() forwards it into a spawn's WIRE_SSH_HOST so the
  // dashboard attach button works from another cockpit; for sudo'd spawns the
  // crew-service overlays its own CREW_SVC_SSH_HOST. [[reference-wireattach-clicktoattach]]
  const WIRE_SSH_HOST = process.env.WIRE_SSH_HOST;
  // THIS machine's crew-service Wire id — every spawn lands there
  // (crew.agent_spawn). Convention: crew-svc@<WIRE_INSTANCE_NAME normalized>.
  // Defaults per instance name if unset; required for spawn to work.
  const CREW_SVC_DEST = process.env.CREW_SVC_DEST;

  if (!AGENT_ID || !rawKey || !WIRE_URL) {
    console.error(
      `[bridge] missing required env: AGENT_ID=${!!AGENT_ID} AGENT_PRIVATE_KEY=${!!rawKey} WIRE_URL=${!!WIRE_URL} — tools will return errors until set`,
    );
  } else if (!CREW_SVC_DEST) {
    console.error(
      `[bridge] missing CREW_SVC_DEST — the crew-service Wire id spawns route to (e.g. crew-svc@patisserie). spawn() will error until set; other tools work.`,
    );
  } else {
    const keypair = await importKeyPair(rawKey);
    const terminalType = detectTerminal();
    const terminal = await createBackend(terminalType);
    const orchestrator = new Orchestrator(terminal);

    // RPC transport for crew.agent_spawn. The bridge holds its OWN receiving
    // WireConnection under the parent identity (a distinct cc_session_id so it
    // coexists with the persona's channel connection); the broker fans the
    // rpc.reply to both sessions, but wire-tools' channel deliver drops rpc.*
    // frames so the persona sees no noise, and this client resolves the reply.
    const pubkey = await derivePublicKeyB64(keypair.privateKey);
    const rpcClient = new RpcClient({ url: WIRE_URL, agentId: AGENT_ID, signingKey: keypair.privateKey });
    const rpcConn = new WireConnection({
      url: WIRE_URL,
      agentId: AGENT_ID,
      agentName: AGENT_ID,
      keyPair: { publicKey: pubkey, privateKey: keypair.privateKey },
      ccSessionId: `bridge-rpc-${AGENT_ID}`,
      deliver: async ({ raw }) => { rpcClient.handleEvent(raw); },
    });
    // grok-wire-bridge already holds the persona's inbound SSE. A second
    // WireConnection under the same AGENT_ID is another dashboard session
    // (Brioche showed 3). Skip it when GROK_WIRE_BRIDGE=1; spawn RPC then
    // uses crew-service HTTP MCP (127.0.0.1:9787) instead.
    if (process.env.GROK_WIRE_BRIDGE === "1") {
      console.error(`[bridge] GROK_WIRE_BRIDGE=1 — skipping extra RPC WireConnection (inbound is the grok-wire-bridge)`);
    } else {
    // Best-effort: if the connection can't open (broker down), spawn's RPC
    // will time out with a clear message — don't crash the whole MCP server.
    try {
      await rpcConn.start();
    } catch (e) {
      console.error(`[bridge] RPC connection failed to start (spawn will error until the broker is reachable): ${(e as Error).message}`);
    }
    }

    deps = {
      orchestrator,
      wire_url: WIRE_URL,
      wire_external_url: WIRE_EXTERNAL_URL,
      wire_ssh_host: WIRE_SSH_HOST,
      parent_agent_id: AGENT_ID,
      parent_signing_key: keypair.privateKey,
      rpc_request: (dest, method, params, timeoutMs) => rpcClient.request(dest, method, params, timeoutMs),
      crew_svc_dest: CREW_SVC_DEST,
    };
    console.error(`[bridge] ready (agent=${AGENT_ID}, backend=${terminalType}, crew_svc=${CREW_SVC_DEST}, wire_external=${WIRE_EXTERNAL_URL ?? "(none, falls back to WIRE_URL)"})`);
  }

}
