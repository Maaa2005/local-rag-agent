'use strict';

// ── State ──────────────────────────────────────────────────
let token = sessionStorage.getItem('token') || '';
let currentUser = null;
let isStreaming = false;
let currentConversationId = null;

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
const passwordBtn     = document.getElementById('password-btn');
const passwordDialog  = document.getElementById('password-dialog');
const passwordForm    = document.getElementById('password-form');
const passwordWarning = document.getElementById('password-warning');
const passwordMsg     = document.getElementById('password-msg');
const passwordCancelBtn = document.getElementById('password-cancel-btn');

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
  if (data.must_change_password) {
    openPasswordDialog(true);
  }
});

logoutBtn.addEventListener('click', logout);

function logout() {
  token = '';
  currentUser = null;
  currentConversationId = null;
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

  loginScreen.hidden = true;
  chatScreen.hidden = false;
  questionInput.focus();
  loadConversations();
}

// ── Password change ──────────────────────────────────────────
function openPasswordDialog(showWarning) {
  passwordMsg.textContent = '';
  passwordMsg.className = 'msg-inline';
  passwordForm.reset();
  passwordWarning.hidden = !showWarning;
  passwordDialog.showModal();
}

passwordBtn.addEventListener('click', () => openPasswordDialog(false));

passwordCancelBtn.addEventListener('click', () => {
  passwordDialog.close();
});

passwordForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const current_password = document.getElementById('current-password').value;
  const new_password = document.getElementById('new-password-input').value;
  const res = await api('POST', '/api/auth/password', { current_password, new_password });
  if (!res) return;
  if (res.ok) {
    passwordMsg.textContent = 'パスワードを変更しました';
    passwordMsg.className = 'msg-inline ok';
    passwordWarning.hidden = true;
    setTimeout(() => passwordDialog.close(), 1200);
  } else {
    const err = await res.json().catch(() => ({}));
    passwordMsg.textContent = err.detail || '変更に失敗しました';
    passwordMsg.className = 'msg-inline err';
  }
});

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

function sourceBasename(source) {
  return (source.source_file || '').split('/').pop() || '(不明)';
}

// ── Citation dialog ────────────────────────────────────────
const citationDialog = document.getElementById('citation-dialog');
const citationDialogBody = document.getElementById('citation-dialog-body');
document.getElementById('citation-close-btn').addEventListener('click', () => {
  citationDialog.close();
});

function openCitationDialog(source) {
  citationDialogBody.innerHTML = '';
  const fileEl = document.createElement('div');
  fileEl.className = 'citation-file';
  fileEl.textContent = sourceBasename(source);
  const contentEl = document.createElement('pre');
  contentEl.className = 'citation-content';
  contentEl.textContent = source.content || '(本文なし)';
  citationDialogBody.appendChild(fileEl);
  citationDialogBody.appendChild(contentEl);
  citationDialog.showModal();
}

function setSources(sourcesRow, sources) {
  sourcesRow.innerHTML = '';
  const seen = new Set();
  (sources || []).forEach((s) => {
    const key = s.source_file || '';
    if (seen.has(key)) return;
    seen.add(key);
    const tag = document.createElement('button');
    tag.type = 'button';
    tag.className = 'source-tag';
    tag.title = key;
    tag.textContent = sourceBasename(s);
    tag.addEventListener('click', () => openCitationDialog(s));
    sourcesRow.appendChild(tag);
  });
}

// 回答テキスト中の [1] [2] ... を、対応する引用元があればクリック可能なチップに変換する。
// テキストは必ず textContent / createElement で挿入し、innerHTML に生文字列を渡さない。
function renderAnswerWithCitations(bubble, text, sources) {
  bubble.innerHTML = '';
  const byId = new Map((sources || []).map((s) => [String(s.id), s]));
  const re = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match;
  while ((match = re.exec(text)) !== null) {
    if (match.index > lastIndex) {
      bubble.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
    }
    const src = byId.get(match[1]);
    if (src) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'citation-chip';
      chip.textContent = `[${match[1]}]`;
      chip.addEventListener('click', () => openCitationDialog(src));
      bubble.appendChild(chip);
    } else {
      bubble.appendChild(document.createTextNode(match[0]));
    }
    lastIndex = re.lastIndex;
  }
  if (lastIndex < text.length) {
    bubble.appendChild(document.createTextNode(text.slice(lastIndex)));
  }
}

// ── Conversation sidebar ───────────────────────────────────
async function loadConversations() {
  const res = await api('GET', '/api/conversations');
  if (!res || !res.ok) return;
  renderConversationList(await res.json());
}

function renderConversationList(rows) {
  const list = document.getElementById('conversation-list');
  list.innerHTML = '';
  rows.forEach((c) => {
    const li = document.createElement('li');
    li.className = 'conversation-item' + (c.id === currentConversationId ? ' active' : '');
    li.dataset.id = String(c.id);

    const titleBtn = document.createElement('button');
    titleBtn.type = 'button';
    titleBtn.className = 'conversation-title';
    titleBtn.textContent = c.title || '(無題)';
    titleBtn.addEventListener('click', () => selectConversation(c.id));

    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.className = 'conversation-delete';
    delBtn.title = '削除';
    delBtn.textContent = '×';
    delBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('この会話を削除しますか？')) return;
      const r = await api('DELETE', `/api/conversations/${c.id}`);
      if (r && r.ok) {
        if (currentConversationId === c.id) startNewConversation();
        loadConversations();
      }
    });

    li.appendChild(titleBtn);
    li.appendChild(delBtn);
    list.appendChild(li);
  });
}

function startNewConversation() {
  currentConversationId = null;
  messages.innerHTML = '';
  document.querySelectorAll('.conversation-item').forEach((li) => li.classList.remove('active'));
}

async function selectConversation(id) {
  if (isStreaming) return;
  const res = await api('GET', `/api/conversations/${id}/messages`);
  if (!res || !res.ok) return;
  const msgs = await res.json();
  currentConversationId = id;
  messages.innerHTML = '';
  msgs.forEach((m) => {
    if (m.role === 'user') {
      appendMessage('user', m.content);
      return;
    }
    const { bubble, sourcesRow, cursor } = appendAssistantShell();
    cursor.remove();
    setSources(sourcesRow, m.sources || []);
    renderAnswerWithCitations(bubble, m.content, m.sources || []);
  });
  document.querySelectorAll('.conversation-item').forEach((li) => {
    li.classList.toggle('active', li.dataset.id === String(id));
  });
}

document.getElementById('new-conversation-btn').addEventListener('click', () => {
  if (isStreaming) return;
  startNewConversation();
});

async function sendQuestion() {
  const q = questionInput.value.trim();
  if (!q || isStreaming) return;

  questionInput.value = '';
  questionInput.style.height = '';
  appendMessage('user', q);
  const { bubble, sourcesRow, cursor } = appendAssistantShell();

  isStreaming = true;
  sendBtn.disabled = true;

  let rawAnswer = '';
  let latestSources = [];
  let hadError = false;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token,
      },
      body: JSON.stringify({ question: q, conversation_id: currentConversationId }),
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

        if (evt.type === 'meta') {
          currentConversationId = evt.conversation_id;
          loadConversations();
        } else if (evt.type === 'sources') {
          latestSources = evt.sources || [];
          setSources(sourcesRow, latestSources);
        } else if (evt.type === 'token') {
          if (!textNode) {
            cursor.remove();
            textNode = document.createTextNode('');
            bubble.appendChild(textNode);
          }
          rawAnswer += evt.content;
          textNode.textContent += evt.content;
          messages.scrollTop = messages.scrollHeight;
        } else if (evt.type === 'error') {
          hadError = true;
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

    if (!hadError && rawAnswer) {
      renderAnswerWithCitations(bubble, rawAnswer, latestSources);
    }
    loadConversations();
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
  await Promise.all([
    loadFolders(),
    loadDocuments(),
    loadTasks(),
    loadUnclassified(),
    loadAuditLogs(),
  ]);
}

async function loadUnclassified() {
  const warningCard = document.getElementById('unclassified-warning');
  const res = await api('GET', '/api/admin/documents/unclassified');
  if (!res || !res.ok) { warningCard.hidden = true; return; }
  const rows = await res.json();
  warningCard.hidden = rows.length === 0;
  const tbody = document.querySelector('#unclassified-table tbody');
  tbody.innerHTML = '';
  rows.forEach((r) => {
    const name = r.source_path.split('/').pop();
    const ts = r.updated_at ? r.updated_at.slice(0, 16).replace('T', ' ') : '-';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td title="${r.source_path}">${name}</td>
      <td>${r.file_type}</td>
      <td class="status-${r.status}">${r.status}</td>
      <td>${ts}</td>`;
    tbody.appendChild(tr);
  });
}

let auditLogCache = [];

async function loadAuditLogs() {
  const res = await api('GET', '/api/admin/audit-logs?limit=50&offset=0');
  if (!res || !res.ok) return;
  auditLogCache = await res.json();
  const tbody = document.querySelector('#audit-table tbody');
  tbody.innerHTML = '';
  auditLogCache.forEach((r, idx) => {
    const ts = r.created_at ? r.created_at.slice(0, 16).replace('T', ' ') : '-';
    const tr = document.createElement('tr');
    tr.dataset.idx = idx;
    tr.innerHTML = `
      <td>${ts}</td>
      <td>${r.username}</td>
      <td>${(r.question || '').slice(0, 40)}</td>
      <td>${(r.retrieved_chunks || []).length}</td>`;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll('tr').forEach((tr) => {
    tr.addEventListener('click', () => openAuditDetail(auditLogCache[Number(tr.dataset.idx)]));
  });
}

function openAuditDetail(entry) {
  const body = document.getElementById('audit-detail-body');
  const sources = (entry.retrieved_chunks || [])
    .map((c) => `${c.source_file ?? '不明'} (score=${c.score ?? '-'}, lv=${c.access_level ?? '-'})`)
    .join('\n') || 'なし';
  body.innerHTML = '';
  const fields = [
    ['日時', entry.created_at || '-'],
    ['ユーザー', entry.username || '-'],
    ['質問', entry.question || '-'],
    ['参照チャンク', sources],
    ['回答', entry.answer || (entry.error ? '(エラーのため回答なし)' : '-')],
    ['エラー', entry.error || 'なし'],
  ];
  fields.forEach(([label, value]) => {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = value;
    body.appendChild(dt);
    body.appendChild(dd);
  });
  document.getElementById('audit-dialog').showModal();
}

document.getElementById('audit-close-btn').addEventListener('click', () => {
  document.getElementById('audit-dialog').close();
});

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
