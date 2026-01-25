/**
 * @fileoverview WebGL Mood Shader for atmospheric background visualization.
 * 
 * Renders a reactive 2D gradient shader behind the main visualization canvas.
 * Responds to musical energy, spectral centroid, novelty, and segment color
 * to create an immersive atmosphere that tracks the music's mood.
 * 
 * @author Remixatron Team
 */

/**
 * Manages WebGL context and shader rendering for the mood background.
 * Uses a fullscreen quad with Simplex Noise modulated by musical features.
 */
export class MoodShader {
    /**
     * Creates a new MoodShader instance.
     * @param {HTMLCanvasElement} canvas - The canvas element to render to.
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = null;
        this.program = null;
        this.vao = null;

        // Uniform locations (cached after compilation)
        this.uniforms = {};

        // Current state for lag smoothing
        this.startTime = performance.now();
    }

    /**
     * Initializes WebGL context and compiles shaders.
     * @returns {boolean} True if initialization succeeded, false otherwise.
     */
    initWebGL() {
        const gl = this.canvas.getContext('webgl2', {
            alpha: false,
            antialias: false,
            depth: false,
            stencil: false,
            preserveDrawingBuffer: false
        });

        if (!gl) {
            console.error('[MoodShader] WebGL2 not supported');
            return false;
        }

        this.gl = gl;

        // Compile shaders
        const vertexShader = this._compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
        const fragmentShader = this._compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);

        if (!vertexShader || !fragmentShader) {
            console.error('[MoodShader] Shader compilation failed');
            return false;
        }

        // Create program
        this.program = gl.createProgram();
        gl.attachShader(this.program, vertexShader);
        gl.attachShader(this.program, fragmentShader);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            console.error('[MoodShader] Program link failed:', gl.getProgramInfoLog(this.program));
            return false;
        }

        // Cache uniform locations
        this.uniforms = {
            u_time: gl.getUniformLocation(this.program, 'u_time'),
            u_resolution: gl.getUniformLocation(this.program, 'u_resolution'),
            u_energy: gl.getUniformLocation(this.program, 'u_energy'),
            u_centroid: gl.getUniformLocation(this.program, 'u_centroid'),
            u_color: gl.getUniformLocation(this.program, 'u_color'),
            u_novelty: gl.getUniformLocation(this.program, 'u_novelty')
        };

        // Create fullscreen quad VAO
        this._createFullscreenQuad();

        console.log('[MoodShader] Initialized successfully');
        return true;
    }

    /**
     * Resizes the WebGL viewport to match canvas dimensions.
     * @param {number} width - Canvas width in CSS pixels.
     * @param {number} height - Canvas height in CSS pixels.
     * @param {number} dpr - Device pixel ratio for high-DPI scaling.
     */
    resize(width, height, dpr) {
        if (!this.gl) return;

        // Scale canvas by DPR for sharp rendering
        const w = Math.floor(width * dpr);
        const h = Math.floor(height * dpr);

        this.canvas.width = w;
        this.canvas.height = h;
        this.gl.viewport(0, 0, w, h);
    }

    /**
     * Renders a single frame of the mood shader.
     * @param {number} time - Current time in seconds (for animation).
     * @param {number} energy - Normalized RMS energy (0.0-1.0).
     * @param {number} centroid - Spectral centroid (0.0-1.0).
     * @param {number[]} color - RGB color array [r, g, b], each 0.0-1.0.
     * @param {number} novelty - Novelty score (0.0-1.0).
     */
    render(time, energy, centroid, color, novelty) {
        const gl = this.gl;
        if (!gl || !this.program) return;

        gl.useProgram(this.program);

        // Set uniforms
        gl.uniform1f(this.uniforms.u_time, time);
        gl.uniform2f(this.uniforms.u_resolution, this.canvas.width, this.canvas.height);
        gl.uniform1f(this.uniforms.u_energy, energy);
        gl.uniform1f(this.uniforms.u_centroid, centroid);
        gl.uniform3fv(this.uniforms.u_color, color);
        gl.uniform1f(this.uniforms.u_novelty, novelty);

        // Draw fullscreen quad
        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.bindVertexArray(null);
    }

    /**
     * Cleans up WebGL resources.
     */
    destroy() {
        if (!this.gl) return;

        if (this.program) {
            this.gl.deleteProgram(this.program);
        }
        if (this.vao) {
            this.gl.deleteVertexArray(this.vao);
        }

        this.gl = null;
        this.program = null;
        this.vao = null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PRIVATE METHODS
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Compiles a GLSL shader from source.
     * @param {number} type - gl.VERTEX_SHADER or gl.FRAGMENT_SHADER.
     * @param {string} source - GLSL source code.
     * @returns {WebGLShader|null} Compiled shader or null on error.
     * @private
     */
    _compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('[MoodShader] Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    /**
     * Creates a fullscreen quad VAO for rendering.
     * @private
     */
    _createFullscreenQuad() {
        const gl = this.gl;

        // Two triangles covering clip space [-1, 1]
        const vertices = new Float32Array([
            -1, -1,
            1, -1,
            1, 1,
            -1, -1,
            1, 1,
            -1, 1
        ]);

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

        gl.bindVertexArray(null);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL SHADER SOURCE CODE
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Vertex shader: Simple passthrough for fullscreen quad.
 */
const VERTEX_SHADER_SRC = `#version 300 es
layout(location = 0) in vec2 a_position;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

/**
 * Fragment shader: Generates atmospheric gradient with Simplex Noise.
 * 
 * Uniforms:
 * - u_time: Animation time in seconds
 * - u_resolution: Canvas resolution in pixels
 * - u_energy: RMS energy (0-1), modulates brightness
 * - u_centroid: Spectral centroid (0-1), modulates noise frequency
 * - u_color: Base RGB color from segment cluster
 * - u_novelty: Novelty score (0-1), triggers flash effects
 */
const FRAGMENT_SHADER_SRC = `#version 300 es
precision highp float;

uniform float u_time;
uniform vec2 u_resolution;
uniform float u_energy;
uniform float u_centroid;
uniform vec3 u_color;
uniform float u_novelty;

out vec4 fragColor;

// ─────────────────────────────────────────────────────────────────────────────
// SIMPLEX NOISE (adapted from Ashima Arts / Ian McEwan)
// ─────────────────────────────────────────────────────────────────────────────

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    
    i = mod289(i);
    vec4 p = permute(permute(permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    
    // Base gradient from segment color (darker at bottom, brighter at top)
    vec3 baseColor = mix(u_color * 0.15, u_color * 0.6, uv.y);
    
    // Noise layer (frequency scales with centroid = "brightness" of sound)
    float noiseScale = 2.0 + u_centroid * 3.0;
    float n = snoise(vec3(uv * noiseScale, u_time * 0.08));
    n = n * 0.5 + 0.5; // Remap from [-1,1] to [0,1]
    
    // Apply noise as subtle variation (reduced from 0.2 to 0.05)
    vec3 noiseColor = u_color * n * 0.05;
    
    // Energy modulates overall brightness (lowered from 0.3+0.7 to 0.1+0.3)
    float brightness = 0.1 + u_energy * 0.3;
    
    // Novelty creates brief "flash" on section changes
    float flash = u_novelty * u_novelty * 0.4;
    
    // Combine
    vec3 finalColor = (baseColor + noiseColor) * brightness + vec3(flash);
    
    // Radial cutout: fade to black in center where rings live
    // smoothstep(inner, outer, dist) -> 0 at center, 1 at edges
    float distFromCenter = length(uv - 0.5);
    float cutout = smoothstep(0.25, 0.6, distFromCenter);
    finalColor *= cutout;
    
    // Note: Vignette removed — cutout already creates dark center / lit edges.
    // The vignette was darkening edges too, leaving nowhere visible.
    
    fragColor = vec4(finalColor, 1.0);
}
`;
