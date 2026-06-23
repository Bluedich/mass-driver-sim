// Preline's own DOMContentLoaded fires before Dash's React tree renders.
// Observe the Dash root container and re-run autoInit once after first render.
(function () {
    function init() {
        if (window.HSStaticMethods) window.HSStaticMethods.autoInit();
    }
    document.addEventListener('DOMContentLoaded', function () {
        var root = document.getElementById('_dash-app-content');
        if (!root) { setTimeout(init, 400); return; }
        var observer = new MutationObserver(function (_, obs) {
            obs.disconnect();
            init();
        });
        observer.observe(root, { childList: true });
    });
}());
