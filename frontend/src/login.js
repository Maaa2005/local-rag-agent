'use strict';

const loginForm = document.getElementById('login-form');
const loginButton = document.getElementById('login-btn');
const loginError = document.getElementById('login-error');

if (new URLSearchParams(window.location.search).get('reason') === 'session_expired') {
  loginError.textContent = 'セッションの有効期限が切れました。もう一度ログインしてください。';
  loginError.hidden = false;
}

async function redirectIfAuthenticated() {
  const existingToken = sessionStorage.getItem('token');
  if (!existingToken) return;
  try {
    const response = await fetch('/api/auth/me', {
      headers: { 'Authorization': 'Bearer ' + existingToken },
    });
    if (response.ok) {
      window.location.replace('/');
      return;
    }
  } catch (_) {
    // ログイン画面を表示して再試行できるようにする。
  }
  sessionStorage.removeItem('token');
}

loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  loginError.hidden = true;
  loginButton.disabled = true;
  loginButton.textContent = 'ログイン中…';
  const form = new FormData();
  form.append('username', document.getElementById('username').value.trim());
  form.append('password', document.getElementById('password').value);
  try {
    const response = await fetch('/api/auth/token', { method: 'POST', body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || 'ログインできませんでした');
    }
    sessionStorage.setItem('token', data.access_token);
    if (data.must_change_password) {
      sessionStorage.setItem('must_change_password', '1');
    } else {
      sessionStorage.removeItem('must_change_password');
    }
    window.location.replace('/');
  } catch (error) {
    loginError.textContent = error.message || 'ネットワークエラーが発生しました';
    loginError.hidden = false;
    document.getElementById('password').value = '';
    document.getElementById('password').focus();
    loginButton.disabled = false;
    loginButton.textContent = 'ログイン';
  }
});

redirectIfAuthenticated();
