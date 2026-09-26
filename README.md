# SPH simulation and rendering created using Orillusion Engine

This is a project created for a bachelor's thesis. The goal is to demonstrate SPH (Smooth Particle Hydrodynamics) in real time graphics.
More focus is given to the physics/simulation side than it is to the rendering side. This is a standalone project made for demostration.
It is not that useful beyond that

## Sources

### Physics and simulation
Ihmsen, M., Orthmann, J., Solenthaler, B., Kolb, A., & Teschner, M. (2014). SPH Fluids in Computer Graphics (Eurographics State-of-the-Art
Report). Eurographics Association. https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf

This is the most used paper in this work. It is the one to explain the physics behind the SPH and also gives the logic which makes the SPH
what it is.

### Rendering
van der Laan, W. J., Green, S., & Sainz, M. (2009). Screen space fluid rendering with curvature flow. In Proceedings of the 2009 Symposium
on Interactive 3D Graphics and Games (pp. 91–98). Association for Computing Machinery. doi.org
https://wstahw.win.tue.nl/edu/2IV06/andrei/particle_rendering/provided/p91-van_der_laan.pdf

This one is used much more loosely compared to the SPH paper. It was used mostly for a high level idea of what to do in order to get a
more visually appealing image compared to a pool of balls that we'd render otherwise.

### Code structure and engine usage
https://www.orillusion.com/en/guide/
https://github.com/Orillusion/orillusion

Mostly the official orillusion engine page as well as it's own internal code.

## How to set up locally

It's quite a simple project. As long as you have npm installed, you should be ready. Just run `npm ci`, and then after that feel free
to run `npm run dev`. Make sure that the browser you use has WebGPU supported. The official orillusion page at the time of writing this
readme said

> The Orillusion engine requires a browser that supports the latest WebGPU standard to run, such as Chrome >= 114. Therefore, there is no
> need to maintain compatibility with older JavaScript syntax, and it is built and published based on ESNext by default. If you use build
> tools such as Vite or Webpack to deploy your project, it is recommended to set the build target to ES2021 or above to ensure that all APIs
> run in their optimal state.

## Deployment

This code is currently deployed via Github pages. You can try it out here: https://31mir.github.io/sph-orillusion-engine-demonstration/.
(same requirements apply for the browser as they do when setting up locally)

## How to use the app

When the link is first opened, a black screen will show. In the top right corner of the screen there should be a menu for selecting particle
count per dimension. The initial particles are rendered in a n * n * n box. This means that if you put for example 100 as your choice, you
will attempt to render 100 * 100 * 100 = 1 000 000 particles. This is currently not a good idea unless you are running a NASA computer or
something.

After clicking start you will be met with the render and a small gui on the top right for controlling some parameters. You can move the camera
by holding left click and moving (rotates around origin of the world) or holding right click and moving (moves through the xy plane).

## Current Benchmarks

I've tested this app on a few devices. Running this on an AMD RX 9060 XT 16GB shows over 180 FPS (real count unknown because the FPS is
capped to the refresh rate of the screen) 40 * 40 * 40 = 64 000 particles. At around 50 * 50 * 50 = 125 000 we get drops below 100 FPS.
Setting the the particle count to 60 * 60 * 60 = 216 000 leaves us at 60 ~ 80 FPS. Attempting max 100 * 100 * 100 = 1 000 0000 gives us
10 ~ 40 FPS. Similar results were given by an NVIDIA RTX 3070. Trying this on a device with integrated graphics requires us to drop the
particle count way low to something like 20 * 20 * 20 = 8 000 or 30 * 30 * 30 = 27 000.

Additional problem is the fact that we use SESPH which has bad stability so we have to cap max delta time to a value that's dependable
on the speed of the fluid and the size of the particle. This means that for big particle count (which results in small particle size)
we have really small max delta time. At big particle counts, not even 200 FPS is enough to satisfy delta time. Hence, at big particle
counts the system becomes slow motion as the frame delta time is greater than the max delta time.

## Known problems

- Running the page on a Linux system might not work. Not sure what causes it.
- Resizing the whole window or forcing the resize of the html canvas by opening dev tools or anything similar will cause the app to crash.
  The app does not support dynamic resizing currently.
- If the framerate is too low, the fluid will look like it's lagging behind the rest of the scene when moving a camera. This is due to the
  fact that the fluid is currently rendered from a post processing function that fetches last frame. I know, that's ridiculous, needs fix.
- Switching tabs will cause the fluid to jitter for a second.
