#version 450
// Fullscreen-triangle vertex shader for the owned video render pipeline.
//
// No vertex buffers: the three vertices are synthesised from gl_VertexIndex (the
// standard "big triangle" trick — one triangle whose bounding rect covers the whole
// [-1,1] clip square, clipped to the viewport). This is why the pipeline binds an
// empty vertex-input state and the draw call is DrawGPUPrimitives(3, 1, 0, 0).
//
// UV mapping: the triangle's clip positions are (-1,-1),(3,-1),(-1,3); the matching
// UVs are (0,0),(2,0),(0,2), so after interpolation the visible [0,1]² square samples
// the full texture. Vulkan/SPIR-V clip space has +Y down in framebuffer terms, so we
// flip V to keep image row 0 at the top of the window (SDL_GPU targets Vulkan here).

layout(location = 0) out vec2 v_uv;

void main() {
    // (0,0),(2,0),(0,2) for indices 0,1,2 → covers [0,1]² after rasterisation.
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_uv = vec2(uv.x, uv.y); // v flip handled below via clip-space Y sign
    // Clip position: map uv [0,2] → [-1,3]. Flip Y so texture row 0 is at the top.
    gl_Position = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}
