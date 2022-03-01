import numpy as np
from gym_fish.envs.entities.underwater_agent import underwater_agent
import moderngl
class mesh:
    def __init__(self,agent:underwater_agent,ctx:moderngl.Context,prog:moderngl.Program) -> None:
        self.set_agent(agent)
        self.ctx = ctx
        self.prog = prog
        self.color = (np.random.rand(),np.random.rand(),np.random.rand())
        self.update_mesh_data()
        self.vbo = self.ctx.buffer(self.get_vertices_data().astype('f4').tobytes())
        self.index_buffer = self.ctx.buffer(self.indices.tobytes())
        self.vao =self.ctx.simple_vertex_array(self.prog,self.vbo,'in_pos','in_normal',index_buffer=self.index_buffer,index_element_size=self.indices.dtype.itemsize)

    def set_agent(self,agent:underwater_agent)->None:
        self._agent = agent
    def update_mesh_data(self):
        if self._agent!=None:
            data = self._agent._dynamics.getRenderData()
            self.pos = np.array(data.pos)
            self.normal = np.array(data.normal)
            self.uv = np.array(data.uv)
            self.indices = np.array(data.indices).astype('uintc')

    def update(self):
        self.update_mesh_data()
        self.vbo.write(self.get_vertices_data().astype('f4').tobytes())
    def get_vertices_data(self):
        return np.dstack([self.pos[:,0],self.pos[:,1],self.pos[:,2], self.normal[:,0],self.normal[:,1],self.normal[:,2]])