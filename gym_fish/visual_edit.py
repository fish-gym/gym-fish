#!/usr/bin/env python3


import ctypes
import math
import os.path
import sys
import pathlib

import PySide2.QtWidgets
import numpy as np
from PySide2 import QtCore, QtWidgets, QtOpenGL,QtGui
from PySide2.QtWidgets import QBoxLayout, QSpacerItem, QWidget

ACTUATOR_TYPE_STRS = ["force", "passive", "servo", "mimic", "acceleration", "locked"]

JOINT_TYPE_STRS = ["invalid","revolute", "continuous", "prismatic", "fixed", "floating", "planar", "ball", "universal"]

try:
    import OpenGL.GL as gl
except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    messageBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "OpenGL sample",
                                       "PyOpenGL must be installed to run this example.",
                                       QtWidgets.QMessageBox.Close)
    messageBox.setDetailedText("Run:\npip install PyOpenGL PyOpenGL_accelerate")
    messageBox.exec_()
    sys.exit(1)

from gym_fish.envs.visualization.camera import  camera
from  gym_fish.envs.lib import pyflare as fl

class visualObject:
    def __init__(self):
        self.color = (1,1,1,1)
        self.data = {"position":[],"normal":[],"indices":[]}
    def update(self):
        pass
class visualSphere(visualObject):
    def __init__(self):
        super(visualSphere, self).__init__()
        x_segements = 50
        y_segements=50
        for y in range(y_segements+1):
            for x in range(x_segements+1):
                xSeg = float(x)/x_segements
                ySeg = float(y)/y_segements
                xPos = math.cos(xSeg*2*math.pi)*math.sin(ySeg*math.pi)
                yPos = math.cos(ySeg*math.pi)
                zPos = math.sin(xSeg*2*math.pi)*math.sin(ySeg*math.pi)
                self.data["position"].append((xPos,yPos,zPos))
                self.data["normal"].append((xPos,yPos,zPos))

        for i in range(y_segements):
            for j in range(x_segements):
                self.data['indices'].append(i * (x_segements + 1) + j)
                self.data['indices'].append((i+1) * (x_segements + 1) + j)
                self.data['indices'].append((i+1)  * (x_segements + 1) + (j+1) )
                self.data['indices'].append(i * (x_segements + 1) + j)
                self.data['indices'].append((i+1) * (x_segements + 1) + (j+1))
                self.data['indices'].append(i * (x_segements + 1) + (j+1) )

        self.data['position'] =  np.array(self.data['position']).astype('float32')
        self.data["normal"] =  np.array(self.data["normal"]).astype('float32')
        self.data["indices"] =  np.array(self.data["indices"]).astype('uint32')


    def update(self):
        pass
class visualMesh(visualObject):
    def __init__(self,meshData:fl.RenderData):
        super(visualMesh, self).__init__()
        self.update(meshData)

    def update(self, meshData):
        self.data['position'] = np.array(meshData.pos).astype('float32')
        self.data['normal'] = np.array(meshData.normal).astype('float32')
        self.data['indices'] = np.array(meshData.indices).astype('uint32')
class bufferForVisualObject:
    def __init__(self,obj:visualObject):
        self.init_buffer(obj)

    def init_buffer(self,obj):
        self.VAO = gl.glGenVertexArrays(1);
        self.PVBO = gl.glGenBuffers(1)
        self.NVBO = gl.glGenBuffers(1)
        self.EBO = gl.glGenBuffers(1)
        self.buffer_data(obj)

    def buffer_data(self,obj):
        gl.glBindVertexArray(self.VAO);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.PVBO);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, obj.data["position"].nbytes, obj.data["position"], gl.GL_DYNAMIC_DRAW);
        gl.glEnableVertexAttribArray(0);
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, obj.data["position"].dtype.itemsize*3, ctypes.c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.NVBO);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, obj.data["normal"].nbytes, obj.data["normal"], gl.GL_DYNAMIC_DRAW);
        gl.glEnableVertexAttribArray(1);
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, obj.data["normal"].dtype.itemsize*3, ctypes.c_void_p(0));

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.EBO);
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, obj.data["indices"].nbytes, obj.data["indices"],  gl.GL_DYNAMIC_DRAW);
        gl.glBindVertexArray(0);



class CameraWidget:
    pass


class GLWidget(QtOpenGL.QGLWidget):
    cam_x_changed = QtCore.Signal(int)
    cam_y_changed = QtCore.Signal(int)
    cam_z_changed = QtCore.Signal(int)
    vertex_code = '''
        attribute vec3 position;
        attribute vec3 normal;
        uniform mat4 model;
        uniform mat3 normal_mat;
        uniform mat4 projection_view;
        uniform vec4 color;

        varying vec4 v_color;
        varying vec3 v_normal;
        varying vec4 v_pos;

    void main()
    {

        v_normal =  normal_mat* normal;
        v_color = color;
        v_pos = projection_view * model*vec4(position,1.0);

       gl_Position = v_pos;
    }
    '''

    fragment_code = '''
    varying vec4 v_pos;
    varying vec4 v_color;
    varying vec3 v_normal;
    // entry point
    void main()
    {
        vec4 light_color = vec4(1,1,1,0.2);
        vec3 light_pos = vec3(1,1,1);
        
        vec3 ambient = vec3(0.2,0.2,0.2);
        
        vec3 norm = normalize(v_normal);
        vec3 lightDir = normalize(light_pos - v_pos.rgb);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = vec3(1,1,1) * (diff * v_color.rgb);
        
        vec3 result = ambient + diffuse;
        gl_FragColor =   vec4(result, 1.0);
    }
    '''

    def __init__(self,skeleton_file:str, parent=None):
        super(GLWidget, self).__init__(parent)

        self.cam = camera()
        self.cam.center=[0.0,0.0,1]
        if skeleton_file.endswith(".json"):
            self.skeleton = fl.skeletonFromJson(skeleton_file, 0)
        elif skeleton_file.endswith(".fbx"):
            self.skeleton = fl.skeletonFromFBX(skeleton_file, 0)
        else:
            print("Input file format not supported. ONlY SUPPORTS json,fbx")
            sys.exit()


        self.selected_jnt_name = self.skeleton.joints[0].name
        self.selected_link_name = self.skeleton.links[0].name

        self.joints = {j.name:j for j in self.skeleton.joints}
        self.links = {l.name:l for l in self.skeleton.links}

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.advance)
        timer.start(20)

    def resizeGL(self, width, height):
        gl.glViewport(0,0,width,height)
        self.cam.window_size=[width,height]


    def advance(self):
        """Used in timer to actually rotate the shape."""

        self.updateGL()

    def set_selected_link_sizex(self,v):
        try:
            self.selected_link.size = np.array([
                float(v),self.selected_link.size[1],  self.selected_link.size[2]])
            self.updateGL()
        except:
            pass
    def set_selected_link_sizey(self,v):
        try:
            self.selected_link.size =np.array( [
                self.selected_link.size[0],float(v),self.selected_link.size[2]])
            self.updateGL()
        except:
            pass
    def set_selected_link_sizez(self,v):
        try:
            self.selected_link.size = np.array([
                self.selected_link.size[0], self.selected_link.size[1], float(v)])
            self.updateGL()
        except:
            pass
    def set_selected_link_mass(self,v):
        try:
            self.selected_link.mass = float(v)
        except:
            pass

    def set_selected_joint_springstiffness(self,v):
        try:
            self.selected_joint.springStiffness = float(v)
        except:
            pass
    @property
    def selected_link(self):
        return self.links[self.selected_link_name]
    @property
    def selected_joint(self):
        return self.joints[self.selected_jnt_name]
    def set_selected_joint_limit_effort(self,v):
        try:
            self.selected_joint.limit.effort = np.array(eval(v))
        except:
            print("Limit effort failed to set")

    def set_selected_joint_use_limit(self,v):
        self.selected_joint.useLimits = (v==QtCore.Qt.CheckState.Checked)

    def set_selected_joint_actuator(self, txt):
        self.selected_joint.actuator = eval("fl.actuatorType." + txt);

    def set_selected_joint_type(self, txt):
        self.selected_joint.type = eval("fl.jointType." + txt);
    def set_selected_joint_limit_lower(self,v):
        try:
            self.selected_joint.limit.lower = np.array(eval(v))
        except:
            print("Limit lower failed to set")
    def set_selected_joint_limit_upper(self,v):
        try:
            self.selected_joint.limit.upper = np.array(eval(v))
        except:
            print("Limit Upper failed to set")
    def set_selected_joint_axis(self, v):
        try:
            self.selected_joint.axis= np.array(eval(v))
        except:
            print("axis failed to set")
    def set_selected_joint_limit_velocity(self, v):
        try:
            self.selected_joint.limit.velocity = np.array(eval(v))
        except:
            print("Limit velocity failed to set")
    # slots
    def set_cam_x(self, v):
        try:
            self.cam.center[0] = float(v)
            self.updateGL()
        except:
            pass

    def set_cam_y(self, v):
        try:
            self.cam.center[1] = float(v)
            self.updateGL()
        except:
            pass
    def set_cam_z(self, v):
        try:
            self.cam.center[2] = float(v)
            self.updateGL()
        except:
            pass

    def initializeGL(self):

        self.program = self.initShaderProgram()

        gl.glUseProgram(self.program)

        self.sphere = visualSphere()
        self.sphere_buffer = bufferForVisualObject(self.sphere)
        mesh = visualMesh(meshData=self.skeleton.getRenderData())
        self.mesh_buffer = bufferForVisualObject(mesh)
    def initShaderProgram(self):
        program = gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        # Set shaders source
        gl.glShaderSource(vertex, self.vertex_code)
        gl.glShaderSource(fragment, self.fragment_code)
        # Compile shaders
        gl.glCompileShader(vertex)
        if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(vertex).decode()
            raise RuntimeError("Vertex shader compilation error: %s", error)
        gl.glCompileShader(fragment)
        if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(fragment).decode()
            print(error)
            raise RuntimeError("Fragment shader compilation error")
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(program))
            raise RuntimeError('Linking error')
        gl.glDetachShader(program, vertex)
        gl.glDetachShader(program, fragment)
        return program

    def distribute_mass_by_size(self):
        toatlMass=10
        totalVol=  0
        for link in self.skeleton.links:
            totalVol = totalVol+ link.size[0]*link.size[1]*link.size[2]
        for link in self.skeleton.links:
            my_size = link.size[0]*link.size[1]*link.size[2]+1e-8
            link.mass = my_size/totalVol*toatlMass


    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
        loc = gl.glGetUniformLocation(self.program, "projection_view")
        gl.glUniformMatrix4fv(loc, 1, False, self.cam.viewProejction.astype('f4'))

        self.skeleton.update(True,True)
        gl.glDisable(gl.GL_BLEND);
        gl.glEnable(gl.GL_DEPTH_TEST);
        gl.glEnable(gl.GL_CULL_FACE);
        gl.glCullFace(gl.GL_BACK);
        mesh = visualMesh(meshData=self.skeleton.getRenderData())
        self.mesh_buffer.buffer_data(mesh)
        self.draw(np.eye(4),(0.5,0.5,0.5,0.5),self.mesh_buffer.VAO,len(mesh.data['indices']))
        gl.glDisable(gl.GL_DEPTH_TEST);
        gl.glDisable(gl.GL_CULL_FACE);
        gl.glEnable(gl.GL_BLEND);
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        for joint in self.skeleton.joints:
            model_mat = joint.node.getWorldTrans()
            model_mat[0, 0] = 0.01
            model_mat[1, 1] = 0.01
            model_mat[2, 2] = 0.01
            if joint.name==self.selected_jnt_name:
                color = (1,0,0,0.4)
            else:
                color = (0,0,0,0)
            if self.selected_jnt_name==None:
                color = (0, 0.4, 0.2, 0.4)
            self.draw(model_mat, color, self.sphere_buffer.VAO, len(self.sphere.data['indices']))
        for link in self.skeleton.links:
            if link.name != self.selected_link_name:
                continue
            model_mat = link.getWorldTrans()
            model_mat[:, 0] = model_mat[:, 0] * link.size[0]/2
            model_mat[:, 1] = model_mat[:, 1] * link.size[1]/2
            model_mat[:, 2] = model_mat[:, 2] * link.size[2]/2
            if link.name==self.selected_link_name:
                color = (1,0,0,0.4)
            else:
                color = (0, 0.2, 0.4, 0.4)
            self.draw(model_mat, color, self.sphere_buffer.VAO, len(self.sphere.data['indices']))

    def draw(self, model_mat,color,VAO,indice_num):
        loc = gl.glGetUniformLocation(self.program, "normal_mat")
        gl.glUniformMatrix3fv(loc, 1, False, np.transpose(np.transpose(np.linalg.pinv(model_mat)))[0:3, 0:3])
        model_mat = np.transpose(model_mat)
        loc = gl.glGetUniformLocation(self.program, "model")
        gl.glUniformMatrix4fv(loc, 1, False, model_mat)
        loc = gl.glGetUniformLocation(self.program, "color")
        gl.glUniform4f(loc, *color)
        gl.glBindVertexArray(VAO)
        gl.glDrawElements(gl.GL_TRIANGLES, indice_num, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)
class AspectRatioWidget(QWidget):
    def __init__(self, widget,aspect_ratio, parent=None):
        super().__init__(parent)
        self.aspect_ratio =aspect_ratio
        self.setLayout(QBoxLayout(QBoxLayout.LeftToRight, self))
        #  add spacer, then widget, then spacer
        self.layout().addItem(QSpacerItem(0, 0))
        self.layout().addWidget(widget)
        self.layout().addItem(QSpacerItem(0, 0))

    def resizeEvent(self, e):
        w = e.size().width()
        h = e.size().height()

        if w / h > self.aspect_ratio:  # too wide
            self.layout().setDirection(QBoxLayout.LeftToRight)
            widget_stretch = h * self.aspect_ratio
            outer_stretch = (w - widget_stretch) / 2 + 0.5
        else:  # too tall
            self.layout().setDirection(QBoxLayout.TopToBottom)
            widget_stretch = w / self.aspect_ratio
            outer_stretch = (h - widget_stretch) / 2 + 0.5

        self.layout().setStretch(0, outer_stretch)
        self.layout().setStretch(1, widget_stretch)
        self.layout().setStretch(2, outer_stretch)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self,skeleton_file:str):
        self.skeleton_file = skeleton_file

        super().__init__()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        self.create_gl_widget()

        self.cam_widget = self.create_camera_widget()

        # set the layout
        central_layout= QtWidgets.QHBoxLayout();

        mid_widget = QtWidgets.QWidget()
        mid_layout =    QtWidgets.QVBoxLayout()
        mid_layout.addWidget(self.glWidgetArea)
        mid_layout.addWidget(self.cam_widget )
        mid_widget.setLayout(mid_layout)

        central_layout.addWidget(mid_widget)
        central_layout.addWidget(self.creat_joint_link_list_widget())

        right_widget = QtWidgets.QWidget()
        right_layout  = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.create_joint_property_widget())
        right_layout.addWidget(self.create_link_property_widget())

        right_widget.setLayout(right_layout)

        central_layout.addWidget(right_widget)

        central_widget.setLayout(central_layout)

        self.setWindowTitle("Edit Fish")
        self.resize(self.glWidget.cam.window_size[0],self.glWidget.cam.window_size[1])

    def create_gl_widget(self):
        self.glWidget = GLWidget(self.skeleton_file)
        self.glWidgetArea = QtWidgets.QScrollArea()
        self.glWidgetArea.setWidget(self.glWidget)
        self.glWidgetArea.setWidgetResizable(True)
        self.glWidgetArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.glWidgetArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.glWidgetArea.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)

        return self.glWidgetArea
    def creat_joint_link_list_widget(self):

        j_l_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        self.jnt_listview = QtWidgets.QListView()
        q_model = QtCore.QStringListModel()
        q_model.setStringList([j.name  for j in self.glWidget.skeleton.joints if j.name.find("intermediate")==-1])

        self.jnt_listview.setModel(q_model)
        def on_jnt_item_clicked(index):
            self.glWidget.selected_jnt_name = self.jnt_listview.selectionModel().selection().indexes()[0].data()
            #
            self.cur_joint_springStiffness_txt.setText(str(self.glWidget.selected_joint.springStiffness))
            self.cur_joint_effort_txt.setText(str(self.glWidget.selected_joint.limit.effort))
            self.cur_joint_lower_txt.setText(str(self.glWidget.selected_joint.limit.lower))
            self.cur_joint_upper_txt.setText(str(self.glWidget.selected_joint.limit.upper))
            self.cur_joint_velocity_txt.setText(str(self.glWidget.selected_joint.limit.velocity))
            self.jnt_property_label.setText("Joint Properties: "+self.glWidget.selected_jnt_name+"\n"+
                                            str(self.glWidget.selected_joint.actuator)+"\n"+
                                            str(self.glWidget.selected_joint.type))
            if self.glWidget.selected_joint.useLimits:
                self.use_limit_widget.setCheckState(QtCore.Qt.CheckState.Checked)
            else:
                self.use_limit_widget.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.actuator_cb.setCurrentIndex(ACTUATOR_TYPE_STRS.index(str(self.glWidget.selected_joint.actuator)[13:]))
            self.type_cb.setCurrentIndex(JOINT_TYPE_STRS.index(str(self.glWidget.selected_joint.type)[10:]))


        self.jnt_listview.selectionModel().selectionChanged.connect(on_jnt_item_clicked)

        self.link_listview = QtWidgets.QListView()
        q_model2 = QtCore.QStringListModel()
        q_model2.setStringList([l.name for l in self.glWidget.skeleton.links])
        self.link_listview.setModel(q_model2)
        def on_link_item_clicked(index):
            self.glWidget.selected_link_name = self.link_listview.selectionModel().selection().indexes()[0].data()
            self.cur_link_mass_txt.setText(str(self.glWidget.selected_link.mass))
            self.cur_link_sizex_txt.setText(str(self.glWidget.selected_link.size[0]))
            self.cur_link_sizey_txt.setText(str(self.glWidget.selected_link.size[1]))
            self.cur_link_sizez_txt.setText(str(self.glWidget.selected_link.size[2]))
            self.link_property_label.setText("Link Properties: " + self.glWidget.selected_link_name)
        self.link_listview.selectionModel().selectionChanged.connect(on_link_item_clicked)

        layout.addWidget(QtWidgets.QLabel("joints:"))
        layout.addWidget(self.jnt_listview)
        layout.addWidget(QtWidgets.QLabel("links:"))
        layout.addWidget(self.link_listview)
        j_l_widget.setLayout(layout)
        return j_l_widget

    def create_link_property_widget(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.link_property_label = QtWidgets.QLabel()
        layout.addWidget(self.link_property_label)

        property_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout()
        self.cur_link_mass_txt = self.create_linetext(None,self.glWidget.set_selected_link_mass)
        self.cur_link_sizex_txt = self.create_linetext(None,self.glWidget.set_selected_link_sizex)
        self.cur_link_sizey_txt = self.create_linetext(None,self.glWidget.set_selected_link_sizey)
        self.cur_link_sizez_txt = self.create_linetext(None,self.glWidget.set_selected_link_sizez)

        self.cur_link_mass_txt.setText(str(self.glWidget.selected_link.mass))
        self.cur_link_sizex_txt.setText(str(self.glWidget.selected_link.size[0]))
        self.cur_link_sizey_txt.setText(str(self.glWidget.selected_link.size[1]))
        self.cur_link_sizez_txt.setText(str(self.glWidget.selected_link.size[2]))
        self.link_property_label.setText("Link Properties: "+self.glWidget.selected_link_name)
        form_layout.addRow(QtWidgets.QLabel("mass"),self.cur_link_mass_txt )
        form_layout.addRow(QtWidgets.QLabel("link_size x"), self.cur_link_sizex_txt)
        form_layout.addRow(QtWidgets.QLabel("link_size y"), self.cur_link_sizey_txt)
        form_layout.addRow(QtWidgets.QLabel("link_size z"), self.cur_link_sizez_txt)
        property_widget.setLayout(form_layout)
        layout.addWidget(property_widget)
        widget.setLayout(layout)
        return widget

    def create_joint_property_widget(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.jnt_property_label = QtWidgets.QLabel()
        layout.addWidget(self.jnt_property_label)

        property_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout()
        self.cur_joint_springStiffness_txt = self.create_linetext(None,self.glWidget.set_selected_joint_springstiffness)
        self.cur_joint_effort_txt = self.create_linetext(None,self.glWidget.set_selected_joint_limit_effort)
        self.cur_joint_lower_txt = self.create_linetext(None,self.glWidget.set_selected_joint_limit_lower)
        self.cur_joint_upper_txt = self.create_linetext(None,self.glWidget.set_selected_joint_limit_upper)
        self.cur_joint_velocity_txt = self.create_linetext(None,self.glWidget.set_selected_joint_limit_velocity)
        #
        self.cur_joint_springStiffness_txt.setText(str(self.glWidget.selected_joint.springStiffness))
        self.cur_joint_effort_txt.setText(str(self.glWidget.selected_joint.limit.effort))
        self.cur_joint_lower_txt.setText(str(self.glWidget.selected_joint.limit.lower))
        self.cur_joint_upper_txt.setText(str(self.glWidget.selected_joint.limit.upper))
        self.cur_joint_velocity_txt.setText(str(self.glWidget.selected_joint.limit.velocity))

        self.jnt_property_label.setText("Joint Properties: " + self.glWidget.selected_jnt_name + "\n" +
                                        str(self.glWidget.selected_joint.actuator) + "\n" +
                                        str(self.glWidget.selected_joint.type))
        def on_select_jnt_actuator(txt):
            self.glWidget.set_selected_joint_actuator(txt)
            self.jnt_property_label.setText("Joint Properties: " + self.glWidget.selected_jnt_name + "\n" +
                                            str(self.glWidget.selected_joint.actuator) + "\n" +
                                            str(self.glWidget.selected_joint.type))
        def on_select_jnt_type(txt):
            self.glWidget.set_selected_joint_type(txt)
            self.jnt_property_label.setText("Joint Properties: " + self.glWidget.selected_jnt_name + "\n" +
                                            str(self.glWidget.selected_joint.actuator) + "\n" +
                                            str(self.glWidget.selected_joint.type))
        self.actuator_cb = QtWidgets.QComboBox()
        self.actuator_cb.addItems(ACTUATOR_TYPE_STRS)
        self.actuator_cb.currentTextChanged.connect(on_select_jnt_actuator)
        self.actuator_cb.setCurrentIndex(int(self.glWidget.selected_joint.actuator))
        self.type_cb = QtWidgets.QComboBox()
        self.type_cb.addItems(JOINT_TYPE_STRS)
        self.type_cb.currentTextChanged.connect(on_select_jnt_type)
        self.type_cb.setCurrentIndex(int(self.glWidget.selected_joint.type))

        layout.addWidget(self.actuator_cb)
        layout.addWidget(self.type_cb)

        form_layout.addRow(QtWidgets.QLabel("spring stiffness"),self.cur_joint_springStiffness_txt )
        self.use_limit_widget = QtWidgets.QCheckBox("Limit: ")
        self.use_limit_widget.stateChanged.connect(self.glWidget.set_selected_joint_use_limit)
        form_layout.addRow(self.use_limit_widget,None)
        form_layout.addRow(QtWidgets.QLabel("limit effort "), self.cur_joint_effort_txt)
        form_layout.addRow(QtWidgets.QLabel("limit lower"), self.cur_joint_lower_txt)
        form_layout.addRow(QtWidgets.QLabel("limit upper"), self.cur_joint_upper_txt)
        form_layout.addRow(QtWidgets.QLabel("limit velocity"), self.cur_joint_velocity_txt)
        property_widget.setLayout(form_layout)
        layout.addWidget(property_widget)
        widget.setLayout(layout)
        return widget


    def create_camera_widget(self):

        x_slider = self.create_linetext(self.glWidget.cam_x_changed,
                                     self.glWidget.set_cam_x)
        y_slider = self.create_linetext(self.glWidget.cam_y_changed,
                                     self.glWidget.set_cam_y)
        z_slider = self.create_linetext(self.glWidget.cam_z_changed,
                                     self.glWidget.set_cam_z)
        x_slider.setText(str(0))
        y_slider.setText(str(0))
        z_slider.setText(str(1))
        cam_widget= QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout()
        self.save_btn =  QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self.save_skeleton)
        form_layout.addRow(self.save_btn)
        self.mass_set_btn =  QtWidgets.QPushButton("Distribute mass by Link Size")
        self.mass_set_btn.clicked.connect(self.glWidget.distribute_mass_by_size)
        form_layout.addRow(self.mass_set_btn)
        form_layout.addRow(QtWidgets.QLabel("Camera pos x"),x_slider)
        form_layout.addRow(QtWidgets.QLabel("Camera pos y"),y_slider)
        form_layout.addRow(QtWidgets.QLabel("Camera pos z"),z_slider)
        cam_widget.setLayout(form_layout)
        return cam_widget
    def save_skeleton(self):
        if(self.glWidget.skeleton!=None):
            fl.SkeletonToJson(self.skeleton_file+"_modified.json",self.glWidget.skeleton)
    def create_linetext(self, changedSignal, setterSlot):
        """Helper to create a slider."""
        line = QtWidgets.QLineEdit()
        # line.setValidator()
        line.textChanged.connect(setterSlot)

        if changedSignal!=None:
            changedSignal.connect(line.setText)

        return line


if __name__ == '__main__':
    if len(sys.argv)>=2:
        file_path = str(sys.argv[1])
    else:
        default_file = str(pathlib.Path(__file__+'/../assets/agents/koi_all_fins.json').resolve())
        print("no json file input, use default :" +default_file)
        file_path =default_file
    if os.path.isfile(file_path):
        skeleton_file =str(pathlib.Path(file_path).resolve())
    else:
        print("Input is not a file")
        sys.exit()
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow(skeleton_file)
    mainWin.show()
    res = app.exec_()
    sys.exit(res)