'use strict';

// ── State ──────────────────────────────────────────────────
let token = sessionStorage.getItem('token') || '';
let currentUser = null;
let isStreaming = false;
let providers = [];        // [{name, display_name, is_external, available, ...}]
let selectedProvider = sessionStorage.getItem('provider') || '';
const consentedProviders = new Set(); // per-session 同意済み外部プロバイダ

// ── DOM refs ───────────────────────────────────────────────
const loginScreen     = document.getElementById('login-screen');
const chatScreen      = document.getElementById('chat-screen');
const loginForm       = document.getElementById('login-form');
const loginError      = document.getElementById('login-error');
const logoutBtn       = document.getElementById('logout-btn');
const userBadge       = document.getElementById('user-badge');
const messages        = document.getElementById('messages');
const questionInput   = document.getElementById('question-input');
const sendBtn         = document.getElementById('send-btn');
const adminNavBtn     = document.getElementById('admin-nav-btn');
const providerSelect  = document.getElementById('provider-select');
const providerBadge   = document.getElementById('provider-badge');
const extModal        = document.getElementById('external-modal');
const extModalName    = document.getElementById('external-modal-name');
const extModalConfirm = document.getElementById('external-modal-confirm');
const extModalCancel  = document.getElementById('external-modal-cancel');

// ── API helper ─────────────────────────────────────────────
async function api(method, path, body) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = 'Bearer ' + token;
  const res = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) { logout(); return null; }
  return res;
}

// ── Auth ───────────────────────────────────────────────────
loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  loginError.hidden = true;
  const form = new FormData();
  form.append('username', document.getElementById('username').value);
  form.append('password', document.getElementById('password').value);
  const res = await fetch('/api/auth/token', { method: 'POST', body: form });
  if (!res.ok) {
    loginError.textContent = 'ユーザー名またはパスワードが正しくありません';
    loginError.hidden = false;
    return;
  }
  const data = await res.json();
  token = data.access_token;
  sessionStorage.setItem('token', token);
  await initChat();
});

logoutBtn.addEventListener('click', logout);

function logout() {
  token = '';
  currentUser = null;
  sessionStorage.removeItem('token');
  chatScreen.hidden = true;
  loginScreen.hidden = false;
}

async function initChat() {
  const res = await api('GET', '/api/auth/me');
  if (!res || !res.ok) { logout(); return; }
  currentUser = await res.json();

  const levelLabel = { 1: 'Lv1 一般', 2: 'Lv2 管理職', 3: 'Lv3 役員' };
  userBadge.textContent = `${currentUser.username}（${levelLabel[currentUser.access_level] ?? 'Lv?'}）`;

  adminNavBtn.style.display = currentUser.access_level >= 3 ? '' : 'none';

  await loadProviders();

  loginScreen.hidden = true;
  chatScreen.hidden = false;
  questionInput.focus();
}

// ── Providers ─────────────────────────────────────────────
async function loadProviders() {
  const res = await api('GET', '/api/providers');
  if (!res || !res.ok) return;
  providers = await res.json();
  renderProviderSelect();
}

function renderProviderSelect() {
  providerSelect.innerHTML = '';
  providers.forEach((p) => {
    const opt = document.createElement('option');
    opt.value = p.name;
    const flag = p.is_external ? ' ☁︎' : ' ⌂';
    const dim  = p.available ? '' : ' (未設定)';
    opt.textContent = p.display_name + flag + dim;
    opt.disabled = !p.available;
    providerSelect.appendChild(opt);
  });

  const fallback = providers.find((p) => p.available)?.name || '';
  if (!selectedProvider || !providers.find((p) => p.name === selectedProvider && p.available)) {
    selectedProvider = fallback;
  }
  if (selectedProvider) {
    providerSelect.value = selectedProvider;
    sessionStorage.setItem('provider', selectedProvider);
  }
  updateProviderBadge();
}

function updateProviderBadge() {
  const p = providers.find((x) => x.name === selectedProvider);
  if (!p) { providerBadge.hidden = true; return; }
  providerBadge.hidden = false;
  providerBadge.textContent = p.is_external ? '外部送信' : 'ローカル';
  providerBadge.className = 'provider-badge ' + (p.is_external ? 'external' : 'local');
}

providerSelect.addEventListener('change', () => {
  selectedProvider = providerSelect.value;
  sessionStorage.setItem('provider', selectedProvider);
  updateProviderBadge();
});

function requestExternalConsent(providerMeta) {
  return new Promise((resolve) => {
    extModalName.textContent = providerMeta.display_name;
    extModal.hidden = false;
    const onConfirm = () => { cleanup(); consentedProviders.add(providerMeta.name); resolve(true); };
    const onCancel  = () => { cleanup(); resolve(false); };
    function cleanup() {
      extModal.hidden = true;
      extModalConfirm.removeEventListener('click', onConfirm);
      extModalCancel.removeEventListener('click', onCancel);
    }
    extModalConfirm.addEventListener('click', onConfirm);
    extModalCancel.addEventListener('click', onCancel);
  });
}

// ── Panel switching ────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach((btn) => {
  btn.addEventListener('click', () => {
    const panel = btn.dataset.panel;
    document.querySelectorAll('.nav-item').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.panel').forEach((p) => {
      const isTarget = p.id === 'panel-' + panel;
      p.hidden = !isTarget;
      p.classList.toggle('active', isTarget);
    });
    if (panel === 'admin') loadAdminData();
  });
});

// ── Chat ───────────────────────────────────────────────────
function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;
  div.appendChild(bubble);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function appendAssistantShell() {
  const div = document.createElement('div');
  div.className = 'msg assistant';

  const sourcesRow = document.createElement('div');
  sourcesRow.className = 'msg-sources';
  div.appendChild(sourcesRow);

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  bubble.appendChild(cursor);
  div.appendChild(bubble);

  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return { bubble, sourcesRow, cursor };
}

function setSources(sourcesRow, sources) {
  sourcesRow.innerHTML = '';
  const unique = [...new Set(sources)];
  unique.forEach((s) => {
    const tag = document.createElement('span');
    tag.className = 'source-tag';
    tag.title = s;
    tag.textContent = s.split('/').pop();
    sourcesRow.appendChild(tag);
  });
}

async function sendQuestion() {
  const q = questionInput.value.trim();
  if (!q || isStreaming) return;

  // 外部プロバイダの場合、未同意ならモーダルで同意を取る
  const meta = providers.find((p) => p.name === selectedProvider);
  if (meta && meta.is_external && !consentedProviders.has(meta.name)) {
    const ok = await requestExternalConsent(meta);
    if (!ok) return;
  }

  questionInput.value = '';
  questionInput.style.height = '';
  appendMessage('user', q);
  const { bubble, sourcesRow, cursor } = appendAssistantShell();

  isStreaming = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token,
      },
      body: JSON.stringify({ question: q, provider: selectedProvider || undefined }),
    });

    if (!res.ok) {
      bubble.textContent = 'エラーが発生しました。再度お試しください。';
      cursor.remove();
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let textNode = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(6)); } catch { continue; }

        if (evt.type === 'sources') {
          setSources(sourcesRow, evt.sources || []);
        } else if (evt.type === 'token') {
          if (!textNode) {
            cursor.remove();
            textNode = document.createTextNode('');
            bubble.appendChild(textNode);
          }
          textNode.textContent += evt.content;
          messages.scrollTop = messages.scrollHeight;
        } else if (evt.type === 'error') {
          cursor.remove();
          const err = document.createElement('div');
          err.className = 'inline-error';
          err.textContent = evt.content;
          bubble.appendChild(err);
        } else if (evt.type === 'done') {
          cursor.remove();
        }
      }
    }
  } catch (err) {
    bubble.textContent = 'ネットワークエラーが発生しました。';
    cursor.remove();
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

sendBtn.addEventListener('click', sendQuestion);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});
questionInput.addEventListener('input', () => {
  questionInput.style.height = '';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + 'px';
});

// ── Admin ──────────────────────────────────────────────────
async function loadAdminData() {
  await Promise.all([loadFolders(), loadDocuments(), loadTasks(), loadProviderCredentials()]);
}

async function loadProviderCredentials() {
  const res = await api('GET', '/api/providers');
  if (!res || !res.ok) return;
  const list = await res.json();
  const container = document.getElementById('provider-cred-list');
  container.innerHTML = '';

  list.forEach((p) => {
    if (!p.requires_credentials) return;
    const card = document.createElement('div');
    card.className = 'cred-row';

    const head = document.createElement('div');
    head.className = 'cred-head';
    head.innerHTML = `<b>${p.display_name}</b> <span class="cred-status ${p.has_credentials ? 'ok' : 'missing'}">${p.has_credentials ? '登録済み' : '未登録'}</span>`;
    card.appendChild(head);

    const form = document.createElement('form');
    form.className = 'cred-form';

    const apiInput = document.createElement('input');
    apiInput.type = 'password';
    apiInput.placeholder = 'API キー (再設定する場合のみ入力)';
    apiInput.dataset.field = 'api_key';
    form.appendChild(apiInput);

    p.extra_fields.forEach((f) => {
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.placeholder = f;
      inp.dataset.field = f;
      form.appendChild(inp);
    });

    const save = document.createElement('button');
    save.type = 'submit';
    save.textContent = '保存';
    form.appendChild(save);

    if (p.has_credentials) {
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn-ghost';
      del.textContent = '削除';
      del.addEventListener('click', async () => {
        if (!confirm(`${p.display_name} の資格情報を削除しますか?`)) return;
        await api('DELETE', `/api/providers/${p.name}/credentials`);
        await loadProviderCredentials();
        await loadProviders();
      });
      form.appendChild(del);
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const apiKey = apiInput.value.trim();
      if (!apiKey) { alert('API キーを入力してください'); return; }
      const extra = {};
      form.querySelectorAll('input[data-field]').forEach((inp) => {
        if (inp.dataset.field === 'api_key') return;
        if (inp.value.trim()) extra[inp.dataset.field] = inp.value.trim();
      });
      const res = await api('PUT', `/api/providers/${p.name}/credentials`, { api_key: apiKey, extra });
      if (res && res.ok) {
        await loadProviderCredentials();
        await loadProviders();
      } else {
        const err = res ? await res.json().catch(() => ({})) : {};
        alert(err.detail || '保存に失敗しました');
      }
    });

    card.appendChild(form);
    container.appendChild(card);
  });
}

async function loadFolders() {
  const res = await api('GET', '/api/admin/folders');
  if (!res || !res.ok) return;
  const rows = await res.json();
  const tbody = document.querySelector('#folder-table tbody');
  tbody.innerHTML = '';
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td title="${r.path}">${r.path}</td>
      <td>${r.access_level}</td>
      <td><button class="delete-btn btn-ghost" data-id="${r.id}">削除</button></td>`;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll('.delete-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      await api('DELETE', `/api/admin/folders/${btn.dataset.id}`);
      loadFolders();
    });
  });
}

async function loadDocuments() {
  const res = await api('GET', '/api/admin/documents');
  if (!res || !res.ok) return;
  const rows = await res.json();
  const tbody = document.querySelector('#doc-table tbody');
  tbody.innerHTML = '';
  rows.forEach((r) => {
    const name = r.source_path.split('/').pop();
    const ts = r.updated_at ? r.updated_at.slice(0, 16).replace('T', ' ') : '-';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td title="${r.source_path}">${name}</td>
      <td>${r.file_type}</td>
      <td class="status-${r.status}">${r.status}</td>
      <td>${r.chunk_count ?? '-'}</td>
      <td>${ts}</td>`;
    tbody.appendChild(tr);
  });
}

async function loadTasks() {
  const res = await api('GET', '/api/admin/tasks');
  if (!res || !res.ok) return;
  const rows = await res.json();
  const tbody = document.querySelector('#task-table tbody');
  tbody.innerHTML = '';
  rows.forEach((r) => {
    const name = (r.source_path || '').split('/').pop();
    const ts = r.updated_at ? r.updated_at.slice(0, 16).replace('T', ' ') : '-';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td title="${r.source_path}">${name}</td>
      <td class="status-${r.status}">${r.status}</td>
      <td>${r.attempts}</td>
      <td>${ts}</td>`;
    tbody.appendChild(tr);
  });
}

document.getElementById('refresh-tasks-btn').addEventListener('click', loadTasks);

document.getElementById('folder-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const path = document.getElementById('folder-path').value.trim();
  const access_level = parseInt(document.getElementById('folder-level').value);
  const res = await api('POST', '/api/admin/folders', { path, access_level });
  if (res && res.ok) {
    document.getElementById('folder-path').value = '';
    loadFolders();
  }
});

document.getElementById('user-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const username = document.getElementById('new-username').value.trim();
  const password = document.getElementById('new-password').value;
  const access_level = parseInt(document.getElementById('new-level').value);
  const msgEl = document.getElementById('user-msg');
  const res = await api('POST', '/api/auth/users', { username, password, access_level });
  if (!res) return;
  if (res.ok) {
    msgEl.textContent = `ユーザー "${username}" を作成しました`;
    msgEl.className = 'msg-inline ok';
    document.getElementById('new-username').value = '';
    document.getElementById('new-password').value = '';
  } else {
    const err = await res.json().catch(() => ({}));
    msgEl.textContent = err.detail || '作成に失敗しました';
    msgEl.className = 'msg-inline err';
  }
});

// ── Boot ───────────────────────────────────────────────────
if (token) {
  initChat();
}
