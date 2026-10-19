// main.js - small helpers used on every page

// Ask for confirmation before dangerous actions (e.g. clearing logs).
document.querySelectorAll("form[data-confirm]").forEach(function (form) {
  form.addEventListener("submit", function (event) {
    if (!window.confirm(form.dataset.confirm)) {
      event.preventDefault();
    }
  });
});
