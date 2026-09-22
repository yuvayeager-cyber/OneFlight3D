import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const viewport = document.querySelector("#viewport");
const status = document.querySelector("#load-status");
const measurementText = document.querySelector("#measurement");
const folderInput = document.querySelector("#folder-input");
const modelInput = document.querySelector("#model-input");
const confidenceButton = document.querySelector("#confidence-toggle");
const measureButton = document.querySelector("#measure-toggle");
const clearButton = document.querySelector("#clear-measurement");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x10141a);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 10_000_000);
camera.position.set(8, 8, 8);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xd9efff, 0x1d2936, 2.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(6, 10, 8);
scene.add(keyLight, new THREE.GridHelper(10, 10, 0x46576d, 0x263140));

let modelRoot = null;
let confidenceRoot = null;
let confidenceVisible = false;
let measurementMode = false;
let selectedPoints = [];
let markers = [];
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.5;
const pointer = new THREE.Vector2();

function resize() {
  const { width, height } = viewport.getBoundingClientRect();
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}
window.addEventListener("resize", resize);
resize();

function disposeObject(object) {
  object.traverse((child) => {
    child.geometry?.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.filter(Boolean).forEach((material) => material.dispose());
  });
}

function clearSceneModel() {
  for (const object of [modelRoot, confidenceRoot]) {
    if (object) { scene.remove(object); disposeObject(object); }
  }
  modelRoot = null;
  confidenceRoot = null;
  confidenceVisible = false;
  confidenceButton.disabled = true;
  confidenceButton.textContent = "Show confidence overlay";
  clearMeasurement();
}

function frameObject(object) {
  const bounds = new THREE.Box3().setFromObject(object);
  if (bounds.isEmpty()) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const distance = Math.max(size.length() * 0.9, 3);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(distance, distance * 0.7, distance));
  camera.near = Math.max(distance / 10_000, 0.01);
  camera.far = distance * 1000;
  camera.updateProjectionMatrix();
  controls.update();
}

function asPoints(geometry) {
  geometry.computeVertexNormals();
  return new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.15, sizeAttenuation: true, vertexColors: true }));
}

function setConfidenceVisibility(show) {
  confidenceVisible = show;
  if (modelRoot) modelRoot.visible = !show || !confidenceRoot;
  if (confidenceRoot) confidenceRoot.visible = show;
  confidenceButton.textContent = show ? "Show textured model" : "Show confidence overlay";
  confidenceButton.classList.toggle("active", show);
}

function basename(file) { return file.webkitRelativePath || file.name; }
function choose(files, endings, excluded = []) {
  return [...files].find((file) => {
    const path = basename(file).toLowerCase();
    return endings.some((ending) => path.endsWith(ending)) && !excluded.some((name) => path.endsWith(name));
  });
}

async function loadPly(file, isConfidence = false) {
  const geometry = new PLYLoader().parse(await file.arrayBuffer());
  const points = asPoints(geometry);
  points.name = isConfidence ? "confidence cloud" : "point cloud";
  return points;
}

async function loadModel(file, files) {
  if (file.name.toLowerCase().endsWith(".ply")) return loadPly(file);
  const manager = new THREE.LoadingManager();
  const fileByPath = new Map();
  for (const candidate of files) {
    fileByPath.set(candidate.name, candidate);
    fileByPath.set(basename(candidate), candidate);
  }
  const objectUrls = [];
  manager.setURLModifier((url) => {
    const name = decodeURIComponent(url).replace(/^.*\//, "");
    const dependent = fileByPath.get(url) || fileByPath.get(name);
    if (!dependent) return url;
    const blobUrl = URL.createObjectURL(dependent);
    objectUrls.push(blobUrl);
    return blobUrl;
  });
  const loader = new GLTFLoader(manager);
  const root = await new Promise((resolve, reject) => loader.load(URL.createObjectURL(file), (gltf) => resolve(gltf.scene), undefined, reject));
  objectUrls.forEach((url) => URL.revokeObjectURL(url));
  return root;
}

async function loadFiles(files) {
  clearSceneModel();
  const modelFile = choose(files, [".glb", ".gltf"]) || choose(files, [".ply"], ["confidence_cloud.ply"]);
  const confidenceFile = choose(files, ["confidence_cloud.ply"]);
  if (!modelFile) {
    status.textContent = "No .glb, .gltf, or .ply file found.";
    return;
  }
  status.textContent = `Loading ${modelFile.name}…`;
  try {
    modelRoot = await loadModel(modelFile, files);
    scene.add(modelRoot);
    if (confidenceFile && confidenceFile !== modelFile) {
      confidenceRoot = await loadPly(confidenceFile, true);
      confidenceRoot.visible = false;
      scene.add(confidenceRoot);
      confidenceButton.disabled = false;
    }
    measureButton.disabled = false;
    clearButton.disabled = false;
    frameObject(modelRoot);
    status.textContent = `Loaded ${modelFile.name}${confidenceRoot ? " with confidence overlay" : ""}.`;
  } catch (error) {
    console.error(error);
    status.textContent = `Could not load ${modelFile.name}: ${error.message || error}`;
  }
}

folderInput.addEventListener("change", (event) => loadFiles(event.target.files));
modelInput.addEventListener("change", (event) => loadFiles(event.target.files));
confidenceButton.addEventListener("click", () => setConfidenceVisibility(!confidenceVisible));
measureButton.addEventListener("click", () => {
  measurementMode = !measurementMode;
  measureButton.classList.toggle("active", measurementMode);
  measureButton.textContent = measurementMode ? "Exit measurement" : "Measure distance";
  measurementText.textContent = measurementMode ? "Measurement: click the first point." : "Measurement: choose two points.";
});

function clearMeasurement() {
  markers.forEach((marker) => scene.remove(marker));
  markers = [];
  selectedPoints = [];
  measurementText.textContent = "Measurement: choose two points.";
}
clearButton.addEventListener("click", clearMeasurement);

renderer.domElement.addEventListener("click", (event) => {
  if (!measurementMode || !(confidenceVisible ? confidenceRoot : modelRoot)) return;
  const bounds = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const activeRoot = confidenceVisible && confidenceRoot ? confidenceRoot : modelRoot;
  const hit = raycaster.intersectObject(activeRoot, true)[0];
  if (!hit) return;
  if (selectedPoints.length === 2) clearMeasurement();
  const point = hit.point.clone();
  selectedPoints.push(point);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffdd57 }));
  marker.position.copy(point); scene.add(marker); markers.push(marker);
  if (selectedPoints.length === 2) {
    const metres = selectedPoints[0].distanceTo(selectedPoints[1]);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(selectedPoints), new THREE.LineBasicMaterial({ color: 0xffdd57 }));
    scene.add(line); markers.push(line);
    measurementText.textContent = `Measurement: ${metres.toFixed(2)} m`;
  } else measurementText.textContent = "Measurement: click the second point.";
});

function render() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(render); }
render();
