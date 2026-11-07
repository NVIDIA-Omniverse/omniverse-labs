/* Native controls remain available when autoplay is blocked or motion is reduced. */
(() => {
  const video = document.querySelector('.blender-hero-video');
  if (!video) return;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const play = () => video.play().catch(() => {});
  const applyMotion = () => {
    video.autoplay = !motion.matches;
    if (motion.matches) video.pause();
    else if (!document.hidden) play();
  };
  motion.addEventListener('change', applyMotion);
  applyMotion();
  let resumeOnReturn = false;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      resumeOnReturn = !video.paused;
      video.pause();
    } else if (resumeOnReturn && !motion.matches) {
      resumeOnReturn = false;
      play();
    }
  });
})();
