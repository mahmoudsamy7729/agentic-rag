(function (window) {
  "use strict";

  var TOKEN_KEY = "agentic_rag_jwt";

  function saveToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
  }

  function getToken() {
    return localStorage.getItem(TOKEN_KEY);
  }

  function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
  }

  function currentPathWithQuery() {
    return window.location.pathname + window.location.search;
  }

  function redirectToLogin(nextPath) {
    var next = nextPath || currentPathWithQuery();
    window.location.href = "/login-ui?next=" + encodeURIComponent(next);
  }

  function requireAuthOrRedirect() {
    var token = getToken();
    if (!token) {
      redirectToLogin();
      return null;
    }
    return token;
  }

  function isAuthEndpoint(url) {
    return (
      String(url).indexOf("/auth/jwt/login") !== -1 ||
      String(url).indexOf("/auth/jwt/refresh") !== -1 ||
      String(url).indexOf("/auth/jwt/logout") !== -1
    );
  }

  async function requestWithBearer(url, options) {
    var opts = options || {};
    var headers = new Headers(opts.headers || {});
    var token = getToken();
    if (token) {
      headers.set("Authorization", "Bearer " + token);
    }

    return fetch(url, {
      ...opts,
      headers: headers,
      credentials: "same-origin",
    });
  }

  async function tryRefreshToken() {
    var response = await fetch("/auth/jwt/refresh", {
      method: "POST",
      credentials: "same-origin",
    });
    if (!response.ok) {
      return false;
    }
    var body = await response.json().catch(function () { return null; });
    if (!body || !body.access_token) {
      return false;
    }
    saveToken(body.access_token);
    return true;
  }

  async function apiFetch(url, options) {
    var opts = options || {};
    var allowRetry = opts._allowRetry !== false;
    var response = await requestWithBearer(url, opts);

    if (response.status === 401) {
      var canRefresh = allowRetry && !isAuthEndpoint(url);
      if (canRefresh) {
        var refreshed = false;
        try {
          refreshed = await tryRefreshToken();
        } catch (_refreshErr) {
          refreshed = false;
        }
        if (refreshed) {
          var retryOptions = { ...opts, _allowRetry: false };
          response = await requestWithBearer(url, retryOptions);
        }
      }

      if (response.status === 401) {
        clearToken();
        redirectToLogin();
        throw new Error("Unauthorized");
      }
    }

    var body = null;
    try {
      body = await response.clone().json();
    } catch (_err) {
      body = null;
    }

    if (!response.ok) {
      if (body && body.detail) {
        throw new Error(String(body.detail));
      }
      throw new Error(response.statusText || "Request failed.");
    }
    return body;
  }

  async function logout() {
    try {
      await fetch("/auth/jwt/logout", {
        method: "POST",
        credentials: "same-origin",
      });
    } catch (_err) {
      // Best effort only for bearer-based auth.
    } finally {
      clearToken();
      redirectToLogin("/login-ui");
    }
  }

  window.uiAuth = {
    saveToken: saveToken,
    getToken: getToken,
    clearToken: clearToken,
    apiFetch: apiFetch,
    requireAuthOrRedirect: requireAuthOrRedirect,
    redirectToLogin: redirectToLogin,
    logout: logout,
  };
})(window);
