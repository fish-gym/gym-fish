import numpy as np 
from .mesh import mesh
from gym_fish.envs.entities.underwater_agent import underwater_agent
from .camera import camera
from PIL import Image
import moderngl


class pointLight:
    def __init__(self,pos=(1,1,1),color=(1.0, 1.0, 1.0, 0.25)) -> None:
        self.pos = pos
        self.color = color
class renderer:
    def __init__(self,camera:camera) -> None:
        self.camera =camera
        self.meshes = []
        self.light = pointLight(tuple
                                (camera.center),(1,1,1,0.25))
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.prog = self.ctx.program(
            vertex_shader='''
            #version 330 core
            in vec3 in_pos;
            in vec3 in_normal;

            out vec3 frag_pos;
            out vec3 frag_norm;

            uniform mat4 mvp;

            void main()
            {
                frag_pos = in_pos;
                frag_norm = in_normal;  
                gl_Position = mvp * vec4(in_pos, 1.0);
            }   
            ''',
    fragment_shader='''
            #version 330 core
                uniform vec3 obj_color;
                uniform vec4 light_color;
                uniform vec3 light_pos;
                in vec3 frag_pos;
                in vec3 frag_norm;
                out vec4 FragColor;
                void main() {
                    float lum = dot(normalize(frag_norm), normalize(frag_pos - light_pos));
                    lum = acos(lum) / 3.14159265;
                    lum = clamp(lum, 0.0, 1.0);
                    lum = lum * lum;
                    lum = smoothstep(0.0, 1.0, lum);
                    lum *= smoothstep(0.0, 80.0, frag_pos.z) * 0.3 + 0.7;
                    lum = lum * 0.8 + 0.2;
                    vec3 color = obj_color;
                    color = color * (1.0 - light_color.a) + light_color.rgb * light_color.a;
                    FragColor = vec4(color * lum, 1.0);
                }
    ''',
        )        

#         self.fbo = self.ctx.simple_framebuffer(self.camera.window_size)
        self.fbo = self.ctx.framebuffer(
        color_attachments= self.ctx.texture(self.camera.window_size, 4),
        depth_attachment=self.ctx.depth_texture(self.camera.window_size)
        )
    def add_mesh(self,agent:underwater_agent):
        self.meshes.append(mesh(agent,self.ctx,self.prog))
    def add_light(self,light:pointLight):
        self.light = light
    def render(self)->Image:
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.prog['light_pos'].value = self.light.pos
        self.prog['light_color'] = self.light.color
        self.prog['mvp'].write(self.camera.viewProejction.astype('f4'))

        for mesh in self.meshes:
            self.prog['obj_color'] = mesh.color
            mesh.update()
            mesh.vao.render(moderngl.TRIANGLES)
        # in the format of bytes
        color = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
        depth = Image.frombytes('L', self.fbo.size, self.fbo.read(attachment=-1), 'raw', 'L').transpose(Image.FLIP_TOP_BOTTOM)
        return (color,depth)
    
    
