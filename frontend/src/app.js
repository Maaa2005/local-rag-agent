'use strict';

// ── State ──────────────────────────────────────────────────
let token = sessionStorage.getItem('token') || '';
let currentUser = null;
let isStreaming = false;

// ── DOM refs ───────────────────────────────────────────────
const loginScreen   = document.getElementById('login-screen');
const chatScreen    = document.getElementById('chat-screen');
const loginForm     = document.getElementById('login-form');
const loginError    = document.getElementById('login-error');
const logoutBtn     = document.getElementById('logout-btn');
const userBadge     = document.getElementById('user-badge');
const messages      = document.getElementById('messages');
const questionInput = document.getElementById('question-input');
const sendBtn       = document.getElementById('send-btn');
const adminNavBtn   = document.getElementById('admin-nav-btn');

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

  // 管理タブはアクセスレベル3のみ表示
  adminNavBtn.style.display = currentUser.access_level >= 3 ? '' : 'none';

  loginScreen.hidden = true;
  chatScreen.hidden = false;
  questionInput.focus();
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
      body: JSON.stringify({ question: q }),
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
  await Promise.all([loadFolders(), loadDocuments(), loadTasks()]);
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
