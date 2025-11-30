// Panorama Viewer using Three.js
let scene, camera, renderer, controls;
let panoramaMesh;

function init() {
    // Create scene
    scene = new THREE.Scene();

    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0.1, 0, 0); // Place camera at center of sphere

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.display = 'block'; // Ensure proper display
    document.getElementById('container').appendChild(renderer.domElement);

    // Add orbit controls for panoramic viewing - use the renderer's canvas element
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.rotateSpeed = 0.5;
    controls.autoRotate = false;  // Disable auto-rotation

    // For a spherical panorama viewer, zoom by changing FOV, not by dollying
    controls.enableZoom = true;  // Enable zoom with scroll wheel
    controls.zoomSpeed = 0.6;    // Zoom speed for FOV changes
    controls.enablePan = false;  // Disable camera movement (panning), we want only rotation
    controls.enableRotate = true; // Enable rotation (this is the "panning" in 360 view)

    // For a 360 panorama sphere, constrain vertical rotation to avoid inside-out view
    controls.minPolarAngle = 0.2; // Prevent looking exactly up
    controls.maxPolarAngle = Math.PI - 0.2; // Prevent looking exactly down

    // Allow horizontal rotation for full 360 degree navigation
    controls.minAzimuthAngle = -Infinity; // Full horizontal rotation
    controls.maxAzimuthAngle = Infinity;  // Full horizontal rotation

    
    // Load panorama texture
    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(
        'panorama.jpg',
        function(texture) {
            // Hide loading message
            document.getElementById('loading').style.display = 'none';

            // Create sphere geometry for equirectangular projection
            const geometry = new THREE.SphereGeometry(500, 60, 40);
            geometry.scale(-1, 1, 1); // Invert the sphere so we can see the inside

            // Create material with loaded texture
            const material = new THREE.MeshBasicMaterial({
                map: texture
            });

            // Create mesh and add to scene
            panoramaMesh = new THREE.Mesh(geometry, material);
            scene.add(panoramaMesh);

            // Start animation loop
            animate();
        },
        undefined, // onProgress callback
        function(err) {
            // Show error message
            document.getElementById('loading').style.display = 'none';
            const errorMsg = document.getElementById('error-message');
            errorMsg.textContent = 'Error loading panorama: Could not load panorama image.';
            errorMsg.style.display = 'block';
            console.error('Error loading panorama:', err);
        }
    );
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);
    
    // Setup UI controls
    setupUIControls();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update(); // Only required if controls.enableDamping = true
    renderer.render(scene, camera);
}

function setupUIControls() {
    // For a spherical panorama viewer, we need to handle zoom differently
    // Since the camera is at the center of the sphere, we use FOV to zoom
    document.getElementById('zoom-in').addEventListener('click', function() {
        // Increase zoom by decreasing FOV
        camera.fov = Math.max(20, camera.fov * 0.9);
        camera.updateProjectionMatrix();
    });

    // Zoom out button
    document.getElementById('zoom-out').addEventListener('click', function() {
        // Decrease zoom by increasing FOV
        camera.fov = Math.min(100, camera.fov * 1.1);
        camera.updateProjectionMatrix();
    });

    // Reset view button
    document.getElementById('reset-view').addEventListener('click', function() {
        // Reset the controls to initial state
        controls.reset();
        // Reset FOV as well
        camera.fov = 75;
        camera.updateProjectionMatrix();
    });

    // Fullscreen button
    document.getElementById('fullscreen').addEventListener('click', function() {
        const container = document.getElementById('container');
        if (container.requestFullscreen) {
            container.requestFullscreen();
        } else if (container.webkitRequestFullscreen) {
            container.webkitRequestFullscreen();
        } else if (container.msRequestFullscreen) {
            container.msRequestFullscreen();
        }
    });
}

// Initialize the viewer when the page loads
window.onload = init;