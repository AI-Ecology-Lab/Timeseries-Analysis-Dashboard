document.addEventListener('DOMContentLoaded', function() {
    const content = document.querySelector('.content');
    const contentRect = content.getBoundingClientRect();
    let mouseX = 0;
    let mouseY = 0;
    let glowX = 0;
    let glowY = 0;

    function updateGlowPosition() {
        // Smoothly interpolate glow position
        glowX += (mouseX - glowX) * 0.1;
        glowY += (mouseY - glowY) * 0.1;

        content.style.setProperty('--mouse-x', `${glowX}px`);
        content.style.setProperty('--mouse-y', `${glowY}px`);

        requestAnimationFrame(updateGlowPosition);
    }

    content.addEventListener('mousemove', function(e) {
        mouseX = e.clientX - contentRect.left;
        mouseY = e.clientY - contentRect.top;
    });

    content.addEventListener('mouseleave', function() {
        mouseX = contentRect.width / 2;
        mouseY = contentRect.height / 2;
    });

    updateGlowPosition();
});