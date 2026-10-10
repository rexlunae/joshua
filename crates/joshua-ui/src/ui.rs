//! Dioxus + Bulma views.
//!
//! Everything the UI shows comes from real server endpoints (`/health`,
//! `/v1/models`, `/v1/chat/completions`) or is measured in the browser
//! (round-trip latency, tokens per second).  Features the server has no API
//! for yet are rendered as clearly labelled placeholders.

use dioxus::core::Task;
use dioxus::prelude::*;
use dioxus_bulma::prelude::*;
use web_time::{Instant, SystemTime, UNIX_EPOCH};

use crate::api::{
    format_utc_hms, normalize_base_url, parse_optional, parse_stop_list, tokens_per_second,
    ApiError, ChatMessage, ChatRequest, Health, ModelList, Role, SamplingParams, Usage,
    DEFAULT_BASE_URL, SERVER_ROUTES,
};
use crate::client::JoshuaClient;

const POLL_INTERVAL_MS: u32 = 5_000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Monitor,
    Playground,
    Cluster,
}

/// Health and model list of one server, as of one poll.
#[derive(Clone, Debug, Default, PartialEq)]
struct NodeStatus {
    health: Option<Result<(Health, f64), ApiError>>,
    models: Option<Result<ModelList, ApiError>>,
    checked_at: Option<u64>,
}

/// Generations sent from this browser session.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SessionStats {
    requests: u32,
    failures: u32,
    completion_tokens: u64,
    last_usage: Option<Usage>,
    last_elapsed_ms: Option<f64>,
    last_tps: Option<f64>,
}

/// Signals shared by every view.
#[derive(Clone, Copy)]
struct AppCtx {
    client: Signal<Option<JoshuaClient>>,
    status: Signal<NodeStatus>,
    stats: Signal<SessionStats>,
    auto_refresh: Signal<bool>,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
async fn sleep_ms(ms: u32) {
    gloo_timers::future::TimeoutFuture::new(ms).await;
}

/// The UI only runs in the browser; host builds exist for `cargo check` and
/// the unit tests, so polling simply never fires there.
#[cfg(not(target_arch = "wasm32"))]
async fn sleep_ms(_ms: u32) {
    std::future::pending::<()>().await;
}

async fn poll_node(client: &JoshuaClient) -> NodeStatus {
    let health = client.health().await;
    let models = client.models().await;
    NodeStatus {
        health: Some(health),
        models: Some(models),
        checked_at: Some(now_unix()),
    }
}

fn refresh(ctx: AppCtx) {
    let mut status = ctx.status;
    if let Some(client) = ctx.client.peek().clone() {
        spawn(async move {
            let s = poll_node(&client).await;
            status.set(s);
        });
    }
}

#[component]
pub fn App() -> Element {
    let ctx = use_context_provider(|| AppCtx {
        client: Signal::new(None),
        status: Signal::new(NodeStatus::default()),
        stats: Signal::new(SessionStats::default()),
        auto_refresh: Signal::new(true),
    });
    let mut view = use_signal(|| View::Monitor);

    // Background poll of the connected server.
    use_future(move || async move {
        loop {
            sleep_ms(POLL_INTERVAL_MS).await;
            if *ctx.auto_refresh.peek() {
                refresh(ctx);
            }
        }
    });

    rsx! {
        document::Title { "joshua-ui" }
        BulmaProvider { theme: BulmaTheme::Auto,
            ConnectionBar {}
            Section {
                Container {
                    Tabs { style: dioxus_bulma::components::TabsStyle::Boxed,
                        Tab { active: view() == View::Monitor, onclick: move |_| view.set(View::Monitor), "Monitor" }
                        Tab { active: view() == View::Playground, onclick: move |_| view.set(View::Playground), "Playground" }
                        Tab { active: view() == View::Cluster, onclick: move |_| view.set(View::Cluster), "Cluster" }
                    }
                    match view() {
                        View::Monitor => rsx! { MonitorView {} },
                        View::Playground => rsx! { PlaygroundView {} },
                        View::Cluster => rsx! { ClusterView {} },
                    }
                }
            }
        }
    }
}

#[component]
fn ConnectionBar() -> Element {
    let ctx = use_context::<AppCtx>();
    let mut url = use_signal(|| DEFAULT_BASE_URL.to_string());
    let mut key = use_signal(String::new);
    let mut error = use_signal(|| None::<String>);

    let connect = move |_| {
        let typed = url.read().clone();
        match normalize_base_url(&typed) {
            Ok(base) => {
                let k = key.read().trim().to_string();
                let client = JoshuaClient::new(base.clone(), (!k.is_empty()).then_some(k));
                let mut c = ctx.client;
                c.set(Some(client));
                let mut s = ctx.status;
                s.set(NodeStatus::default());
                url.set(base);
                error.set(None);
                refresh(ctx);
            }
            Err(e) => error.set(Some(e.to_string())),
        }
    };

    let connected = ctx.client.read().as_ref().map(|c| c.base().to_string());
    let health_tag = match &ctx.status.read().health {
        Some(Ok((h, _))) if h.is_ok() => rsx! { Tag { color: BulmaColor::Success, "healthy" } },
        Some(Ok((h, _))) => rsx! { Tag { color: BulmaColor::Warning, "{h.status}" } },
        Some(Err(_)) => rsx! { Tag { color: BulmaColor::Danger, "unreachable" } },
        None if connected.is_some() => rsx! { Tag { "checking…" } },
        None => rsx! { Tag { "not connected" } },
    };

    rsx! {
        Navbar { color: BulmaColor::Dark,
            NavbarBrand {
                NavbarItem { strong { "joshua-ui" } }
            }
            NavbarMenu { active: true,
                NavbarEnd {
                    NavbarItem {
                        Field { addons: true,
                            Control {
                                Input {
                                    value: url(),
                                    placeholder: "http://127.0.0.1:8080",
                                    oninput: move |e: FormEvent| url.set(e.value()),
                                }
                            }
                            Control {
                                Input {
                                    input_type: InputType::Password,
                                    value: key(),
                                    placeholder: "API key (optional)",
                                    oninput: move |e: FormEvent| key.set(e.value()),
                                }
                            }
                            Control {
                                Button { color: BulmaColor::Primary, onclick: connect, "Connect" }
                            }
                        }
                    }
                    NavbarItem { {health_tag} }
                }
            }
        }
        if let Some(e) = error() {
            Notification { color: BulmaColor::Danger, light: true, "{e}" }
        }
    }
}

fn error_text(e: &ApiError) -> String {
    match e {
        ApiError::Unauthorized(_) => {
            format!("{e} — the server was started with --api-key; enter it above.")
        }
        ApiError::Transport(_) => format!(
            "{e} — is `joshua serve` running at this address and reachable from the browser?"
        ),
        _ => e.to_string(),
    }
}

#[component]
fn MonitorView() -> Element {
    let ctx = use_context::<AppCtx>();
    let mut auto = ctx.auto_refresh;
    let Some(client) = ctx.client.read().clone() else {
        return rsx! {
            Notification { color: BulmaColor::Info, light: true,
                "Enter the base URL of a joshua server (`joshua serve`) and press Connect."
            }
        };
    };
    let status = ctx.status.read().clone();
    let stats = *ctx.stats.read();
    let checked = status
        .checked_at
        .map(format_utc_hms)
        .unwrap_or_else(|| "never".into());

    let health_body = match &status.health {
        None => rsx! { p { "Checking…" } },
        Some(Ok((h, rtt))) => rsx! {
            p { "Status: " strong { "{h.status}" } }
            p { "Round-trip latency: {rtt:.1} ms" }
        },
        Some(Err(e)) => {
            rsx! { Notification { color: BulmaColor::Danger, light: true, "{error_text(e)}" } }
        }
    };

    let models_body = match &status.models {
        None => rsx! { p { "Loading…" } },
        Some(Err(e)) => {
            rsx! { Notification { color: BulmaColor::Danger, light: true, "{error_text(e)}" } }
        }
        Some(Ok(list)) if list.data.is_empty() => rsx! { p { "The server reports no models." } },
        Some(Ok(list)) => rsx! {
            Table { fullwidth: true, striped: true, narrow: true,
                thead { tr { th { "Model id" } th { "Owner" } } }
                tbody {
                    for m in list.data.iter() {
                        tr { key: "{m.id}", td { code { "{m.id}" } } td { "{m.owned_by}" } }
                    }
                }
            }
            p { class: "help",
                "joshua serves one chat model per process; a second entry is the Whisper model when one is loaded."
            }
        },
    };

    let fmt_opt = |v: Option<f64>, unit: &str| {
        v.map(|x| format!("{x:.1} {unit}"))
            .unwrap_or_else(|| "–".into())
    };
    let last_tokens = stats
        .last_usage
        .map(|u| {
            format!(
                "{} prompt + {} completion",
                u.prompt_tokens, u.completion_tokens
            )
        })
        .unwrap_or_else(|| "–".into());

    rsx! {
        Level {
            LevelLeft {
                LevelItem { p { "Server " code { "{client.base()}" } " — last checked {checked}" } }
            }
            LevelRight {
                LevelItem {
                    Checkbox {
                        checked: auto(),
                        onchange: move |e: FormEvent| auto.set(e.checked()),
                        "Auto-refresh every {POLL_INTERVAL_MS / 1000}s"
                    }
                }
                LevelItem { Button { size: BulmaSize::Small, onclick: move |_| refresh(ctx), "Refresh now" } }
            }
        }
        Columns { multiline: true,
            Column { Card {
                CardHeader { CardHeaderTitle { "Health (GET /health)" } }
                CardContent { {health_body} }
            } }
            Column { Card {
                CardHeader { CardHeaderTitle { "Loaded models (GET /v1/models)" } }
                CardContent { {models_body} }
            } }
        }
        Columns {
            Column { Card {
                CardHeader { CardHeaderTitle { "This browser's generations" } }
                CardContent {
                    p { class: "help", "Measured client-side from playground requests; the server has no metrics endpoint." }
                    Table { narrow: true,
                        tbody {
                            tr { th { "Requests" } td { "{stats.requests} ({stats.failures} failed)" } }
                            tr { th { "Completion tokens" } td { "{stats.completion_tokens}" } }
                            tr { th { "Last request tokens" } td { "{last_tokens}" } }
                            tr { th { "Last wall time" } td { "{fmt_opt(stats.last_elapsed_ms, \"ms\")}" } }
                            tr { th { "Last throughput" } td { "{fmt_opt(stats.last_tps, \"tok/s\")}" } }
                        }
                    }
                }
            } }
            Column { Card {
                CardHeader { CardHeaderTitle { "Server endpoints" } }
                CardContent {
                    Table { narrow: true, fullwidth: true,
                        tbody {
                            for (method, path, what) in SERVER_ROUTES.iter() {
                                tr { key: "{path}", td { Tag { "{method}" } } td { code { "{path}" } } td { "{what}" } }
                            }
                        }
                    }
                }
            } }
        }
        Notification { color: BulmaColor::Warning, light: true,
            strong { "Not yet available: " }
            "server-side metrics (queue depth, KV-cache use, per-backend placement, decode tok/s) and runtime controls (load/unload model, change placement). "
            "The joshua server exposes no API for these yet, so this panel will fill in once it does."
        }
    }
}

#[component]
fn PlaygroundView() -> Element {
    let ctx = use_context::<AppCtx>();
    let mut history = use_signal(Vec::<ChatMessage>::new);
    let mut pending = use_signal(String::new);
    let mut input = use_signal(String::new);
    let mut system = use_signal(String::new);
    let mut model = use_signal(String::new);
    let mut max_tokens = use_signal(|| "256".to_string());
    let mut temperature = use_signal(String::new);
    let mut top_p = use_signal(String::new);
    let mut top_k = use_signal(String::new);
    let mut stop = use_signal(String::new);
    let mut stream = use_signal(|| true);
    let mut error = use_signal(|| None::<String>);
    let mut running = use_signal(|| None::<Task>);

    let models: Vec<String> = match &ctx.status.read().models {
        Some(Ok(list)) => list.data.iter().map(|m| m.id.clone()).collect(),
        _ => Vec::new(),
    };
    let first_model = models.first().cloned();
    // The server answers with its loaded model whatever id is sent, so fall
    // back to the first listed one rather than blocking on a selection.
    let effective_model = {
        let m = model.read().clone();
        if m.is_empty() {
            first_model.clone().unwrap_or_else(|| "joshua".into())
        } else {
            m
        }
    };

    let send_model = effective_model.clone();
    let send = move |_| {
        let Some(client) = ctx.client.peek().clone() else {
            error.set(Some("Connect to a server first.".into()));
            return;
        };
        let text = input.peek().trim().to_string();
        if text.is_empty() || running.peek().is_some() {
            return;
        }
        let params = (|| -> Result<SamplingParams, String> {
            Ok(SamplingParams {
                max_tokens: parse_optional(&max_tokens.peek(), "max_tokens")?,
                temperature: parse_optional(&temperature.peek(), "temperature")?,
                top_p: parse_optional(&top_p.peek(), "top_p")?,
                top_k: parse_optional(&top_k.peek(), "top_k")?,
                stop: parse_stop_list(&stop.peek()),
                ..Default::default()
            })
        })();
        let params = match params {
            Ok(p) => p,
            Err(e) => {
                error.set(Some(e));
                return;
            }
        };
        error.set(None);
        history.write().push(ChatMessage::new(Role::User, text));
        input.set(String::new());
        pending.set(String::new());
        let req = ChatRequest::new(
            send_model.clone(),
            Some(&system.peek()),
            &history.peek(),
            &params,
            *stream.peek(),
        );
        let mut stats = ctx.stats;
        let task = spawn(async move {
            let start = Instant::now();
            let result = if req.stream {
                client
                    .chat_stream(&req, |delta| pending.write().push_str(delta))
                    .await
                    .map(|s| (s.text, s.usage, s.tool_calls))
            } else {
                client.chat(&req).await.map(|r| {
                    (
                        r.text().to_string(),
                        Some(r.usage),
                        r.choices.iter().any(|c| c.message.tool_calls.is_some()),
                    )
                })
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let mut st = stats.write();
            st.requests += 1;
            match result {
                Ok((text, usage, tool_calls)) => {
                    let mut text = text;
                    if tool_calls {
                        text.push_str("\n[the model returned tool calls, which the playground does not render]");
                    }
                    history
                        .write()
                        .push(ChatMessage::new(Role::Assistant, text));
                    st.last_elapsed_ms = Some(elapsed);
                    st.last_usage = usage;
                    if let Some(u) = usage {
                        st.completion_tokens += u64::from(u.completion_tokens);
                        st.last_tps = tokens_per_second(u.completion_tokens, elapsed);
                    }
                }
                Err(e) => {
                    st.failures += 1;
                    error.set(Some(error_text(&e)));
                }
            }
            pending.set(String::new());
            running.set(None);
        });
        running.set(Some(task));
    };

    let cancel = move |_| {
        if let Some(t) = running.take() {
            t.cancel();
            let partial = pending.take();
            if !partial.is_empty() {
                history.write().push(ChatMessage::new(
                    Role::Assistant,
                    format!("{partial}\n[cancelled]"),
                ));
            }
        }
    };

    let busy = running.read().is_some();

    rsx! {
        Columns {
            Column { size: dioxus_bulma::components::ColumnSize::OneThird,
                BulmaBox {
                    Field {
                        FieldLabel { "Model" }
                        Control {
                            Select {
                                value: effective_model.clone(),
                                onchange: move |e: FormEvent| model.set(e.value()),
                                if models.is_empty() {
                                    option { value: "{effective_model}", "{effective_model}" }
                                }
                                for m in models.iter() {
                                    option { key: "{m}", value: "{m}", selected: *m == effective_model, "{m}" }
                                }
                            }
                        }
                    }
                    Field {
                        FieldLabel { "System prompt" }
                        Control { Textarea { rows: 3, value: system(), oninput: move |e: FormEvent| system.set(e.value()) } }
                    }
                    Field {
                        FieldLabel { "max_tokens" }
                        Control { Input { value: max_tokens(), oninput: move |e: FormEvent| max_tokens.set(e.value()) } }
                    }
                    Field {
                        FieldLabel { "temperature" }
                        Control { Input { value: temperature(), placeholder: "server default", oninput: move |e: FormEvent| temperature.set(e.value()) } }
                    }
                    Field {
                        FieldLabel { "top_p" }
                        Control { Input { value: top_p(), placeholder: "server default", oninput: move |e: FormEvent| top_p.set(e.value()) } }
                    }
                    Field {
                        FieldLabel { "top_k" }
                        Control { Input { value: top_k(), placeholder: "server default", oninput: move |e: FormEvent| top_k.set(e.value()) } }
                    }
                    Field {
                        FieldLabel { "Stop sequences (comma separated)" }
                        Control { Input { value: stop(), oninput: move |e: FormEvent| stop.set(e.value()) } }
                    }
                    Field {
                        Checkbox { checked: stream(), onchange: move |e: FormEvent| stream.set(e.checked()), "Stream (SSE)" }
                    }
                }
            }
            Column {
                BulmaBox {
                    for (i, m) in history.read().iter().enumerate() {
                        div { key: "{i}", class: "mb-3",
                            Tag { color: if m.role == Role::User { BulmaColor::Info } else { BulmaColor::Primary }, "{m.role.as_str()}" }
                            pre { style: "white-space: pre-wrap;", "{m.content}" }
                        }
                    }
                    if busy {
                        div { class: "mb-3",
                            Tag { color: BulmaColor::Primary, "assistant" }
                            pre { style: "white-space: pre-wrap;", "{pending}" }
                            p { class: "help",
                                "joshua finishes generating before it starts streaming, so the first token may take a while to appear."
                            }
                        }
                    }
                    if history.read().is_empty() && !busy {
                        p { class: "has-text-grey", "No messages yet." }
                    }
                }
                if let Some(e) = error() {
                    Notification { color: BulmaColor::Danger, light: true, "{e}" }
                }
                Field {
                    Control {
                        Textarea {
                            rows: 3,
                            value: input(),
                            placeholder: "Message…",
                            oninput: move |e: FormEvent| input.set(e.value()),
                        }
                    }
                }
                Buttons {
                    Button { color: BulmaColor::Primary, loading: busy, disabled: busy, onclick: send, "Send" }
                    Button { disabled: !busy, onclick: cancel, "Cancel" }
                    Button {
                        disabled: busy,
                        onclick: move |_| { history.write().clear(); error.set(None); },
                        "Clear"
                    }
                }
            }
        }
    }
}

#[derive(Clone, PartialEq)]
struct ClusterNode {
    base: String,
    status: NodeStatus,
}

#[component]
fn ClusterView() -> Element {
    let ctx = use_context::<AppCtx>();
    let mut nodes = use_signal(Vec::<ClusterNode>::new);
    let mut new_url = use_signal(String::new);
    let mut error = use_signal(|| None::<String>);

    let check_all = move || {
        let key = ctx
            .client
            .peek()
            .as_ref()
            .and_then(|c| c.api_key().map(String::from));
        let bases: Vec<String> = nodes.peek().iter().map(|n| n.base.clone()).collect();
        for base in bases {
            let client = JoshuaClient::new(base.clone(), key.clone());
            spawn(async move {
                let s = poll_node(&client).await;
                if let Some(n) = nodes.write().iter_mut().find(|n| n.base == base) {
                    n.status = s;
                }
            });
        }
    };

    let add = move |_| {
        let typed = new_url.read().clone();
        match normalize_base_url(&typed) {
            Ok(base) => {
                if !nodes.peek().iter().any(|n| n.base == base) {
                    nodes.write().push(ClusterNode {
                        base,
                        status: NodeStatus::default(),
                    });
                }
                new_url.set(String::new());
                error.set(None);
                check_all();
            }
            Err(e) => error.set(Some(e.to_string())),
        }
    };

    rsx! {
        Notification { color: BulmaColor::Warning, light: true,
            strong { "Placeholder: cluster topology. " }
            "Clusters run through the one-shot `joshua cluster-run` CLI (built with --features distributed; explicit --peers, rank and world size); "
            "there is no coordinator HTTP endpoint that reports peers, ranks, shard placement or collective health, "
            "so this view cannot show them yet. Until one exists, add each node's `joshua serve` address below to watch "
            "its health and loaded model."
        }
        Field { addons: true,
            Control { expanded: true,
                Input { value: new_url(), placeholder: "http://node2.lan:8080", oninput: move |e: FormEvent| new_url.set(e.value()) }
            }
            Control { Button { color: BulmaColor::Primary, onclick: add, "Add node" } }
            Control { Button { onclick: move |_| check_all(), "Check all" } }
        }
        if let Some(e) = error() {
            Notification { color: BulmaColor::Danger, light: true, "{e}" }
        }
        p { class: "help mb-3", "Nodes are checked with the API key from the connection bar (if any); /health never needs one." }
        Table { fullwidth: true, striped: true,
            thead { tr { th { "Node" } th { "Health" } th { "RTT" } th { "Models" } th { "Checked" } th {} } }
            tbody {
                for node in nodes.read().iter().cloned() {
                    tr { key: "{node.base}",
                        td { code { "{node.base}" } }
                        td {
                            match &node.status.health {
                                Some(Ok((h, _))) if h.is_ok() => rsx! { Tag { color: BulmaColor::Success, "ok" } },
                                Some(Ok((h, _))) => rsx! { Tag { color: BulmaColor::Warning, "{h.status}" } },
                                Some(Err(e)) => rsx! { span { title: "{e}", Tag { color: BulmaColor::Danger, "down" } } },
                                None => rsx! { Tag { "…" } },
                            }
                        }
                        td {
                            match &node.status.health {
                                Some(Ok((_, rtt))) => format!("{rtt:.1} ms"),
                                _ => "–".to_string(),
                            }
                        }
                        td {
                            match &node.status.models {
                                Some(Ok(l)) => l.data.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", "),
                                Some(Err(e)) => e.to_string(),
                                None => "–".to_string(),
                            }
                        }
                        td { {node.status.checked_at.map(format_utc_hms).unwrap_or_else(|| "–".into())} }
                        td {
                            Delete {
                                onclick: {
                                    let base = node.base.clone();
                                    move |_| nodes.write().retain(|n| n.base != base)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
