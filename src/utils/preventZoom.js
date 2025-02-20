export function preventZoom() {
    // Prevent zoom on desktop
    document.addEventListener('wheel', function(e) {
        if(e.ctrlKey) {
            e.preventDefault();
        }
    }, { passive: false });

    // Prevent zoom on mobile/tablet
    document.addEventListener('touchmove', function(e) {
        if (e.scale !== 1) {
            e.preventDefault();
        }
    }, { passive: false });

    // Prevent zoom via keyboard
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && (e.key === '+' || e.key === '-' || e.key === '=')) {
            e.preventDefault();
        }
    });
}