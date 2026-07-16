"""
Tkinter-based object visualiser for the ambisonics player GUI.

The module provides a lightweight helper class for drawing a 3D mesh
from a Wavefront `.obj` file into a Tk canvas. It is used to visualise the
current head orientation in the GUI.
This code is derived from the GitHub repository https://github.com/ES-Alexander/3D-Rendering-Desktop-App.git
and was adapted to work specifically with the ambisonics player. Copyright belongs to their respective owner(s).
"""

import numpy as np
from collections import namedtuple
from tkinter import filedialog, messagebox

from playamb.utils.utils import resolve_path

Angle = namedtuple('Angle', 'x y z')
""" Named tuple holding roll, pitch, and yaw cosine/sine components. """

class Visual3D:
    """ A drawable `.obj` mesh representation for Tk canvas rendering. """
    # .obj format specifications
    VERTEX, FACE = 'v', 'f'
    RELEVANTS = [relevant + ' ' for relevant in (VERTEX, FACE)]

    def __init__(self, filename, canvas, scale=500, position=[]*2,
                 rotation=None, zoom=10, line_colour='#0000FF', point_size=2,
                 point_colour='#FFFFFF'):
        """
        Create a new Visual3D instance.
        
        Parameters
        ----------
        filename : str
            The path to the `.obj` file.
        canvas : tk.Canvas
            The tkinter canvas to draw the 3D object on.
        scale : int, optional
            The scaling factor.
        position, optional : list of int
            Horizontal and vertical offsets to move the object across the canvas.
        rotation, optional : list of int
            Rotation values around the fixed-world axes X, Y and Z.
        zoom, optional : int
            Zooming factor.
        line_colour : 
        """

        self._read_points(filename)
        self._init_display(canvas, scale, position, rotation, zoom,
                           line_colour, point_size, point_colour)

    def _read_points(self, filename):
        """ Parse a `.obj` file to extract the relevant data. """
        # initialise variables
        self.long_filename = self.filename = filename
        if '/' in filename:
            self.filename = self.filename[self.filename.rfind('/')+1:]
        elif '\\' in filename:
            self.filename = self.filename[self.filename.rfind('\\')+1:]

        self._faces    = []
        vertices       = []

        # populate from file
        with open(resolve_path(filename)) as file:
            for line in file:
                # use a generator to stop checking line early if possible
                if all(not line.startswith(relevant)
                       for relevant in self.RELEVANTS):
                    continue # ignore all lines that aren't relevant
                data = line.split()
                line_type = data[0]
                data = data[1:]
                if line_type == self.VERTEX:
                    vertices.append([float(x) for x in data])
                elif line_type == self.FACE:
                    # -1 to switch from .obj indices to 0-based indices
                    self._faces.append([int(f.split('/')[0])-1 for f in data])
            self._vertices = np.array(vertices).T

    def _init_display(self, canvas, scale, position, rotation, zoom,
                      line_colour, point_size, point_colour):
        """ Initialise the display and draw the object. """
        self._position = np.array(position) if position else np.array([420, 540])#[0]*2)
        self._rotation = np.array(rotation) if rotation else np.array([0.]*3)
        self._scale    = scale if scale else 2500 # TODO scale to canvas?
        self._zoom     = zoom if zoom else 20

        self._canvas       = canvas
        self._line_colour  = line_colour
        self._point_size   = point_size
        self._point_colour = point_colour

        self._changed = True
        self._polygon_ids = []
        self.draw(rotation=self._rotation)

    def draw(self, zoom=None, rotation=None, spin=False):
        """ Draw self, zoomed and rotated. """
        # update only if changed since last time
        delta = np.max(np.abs(np.array(rotation) - self._rotation))
        if zoom is not None and zoom != self._zoom:
            self._zoom = zoom
            self._changed = True
        if spin:
            if rotation is not None:
                self._rotation += rotation
            else:
                self._rotation *= 2
            self._changed = True
        elif rotation is not None and delta>0.2: # 0.2 degree threshold on any axis
            self._rotation = np.radians(rotation)
            self._changed = True
        if not self._changed:
            return # change below the threshold, so no need to re-draw

        # only calculate this once, since it's the same for all points
        rotation = self._rotation_matrix()
        C = np.array([
            [0, -1, 0],
            [0,  0, 1],
            [1,  0, 0]
        ])
        rotation = C @ rotation @ C.T
        # vectorised matrix arithmetic - do all points at once
        rotated = rotation @ self._vertices
        point_scales = self._scale / (self._zoom - rotated[2])

        self._projected_points = (rotated[:2] * [[1],[-1]] * point_scales).T \
                                + self._position
        
        if not self._polygon_ids:
            self._create_faces()
        else:
            self._update_faces()

        self._changed = False

    def _rotation_matrix(self):
        """ Returns the current net rotation matrix for self. """
        yaw, pitch, roll = self._rotation
        # match the sign convention of the visualiser
        yaw = -yaw

        cos = Angle(np.cos(roll), np.cos(pitch), np.cos(yaw))
        sin = Angle(np.sin(roll), np.sin(pitch), np.sin(yaw))

        rot_x = np.array([[1, 0    , 0     ],
                            [0, cos.x, -sin.x],
                            [0, sin.x, cos.x ]])

        rot_y = np.array([[cos.y, 0, -sin.y],
                            [0    , 1, 0     ],
                            [sin.y, 0, cos.y ]])

        rot_z = np.array([[cos.z, -sin.z, 0],
                            [sin.z, cos.z , 0],
                            [0    , 0     , 1]])
        return rot_z @ rot_y @ rot_x

    def _draw_projected_points(self):
        """ Draw the current projected points. """
        for point in self._projected_points:
            self._canvas.create_line(*point, *point, width=self._point_size,
                                     fill=self._point_colour)

    def _draw_faces(self):
        """ Draw the lines for each stored face. Superseded by _create_faces() and _update_faces()"""
        for face in self._faces:
            draw_points = []
            for point_index in face:
                draw_points.extend(self._projected_points[point_index])
            self._canvas.create_polygon(draw_points, outline=self._line_colour,
                                        fill='')
    
    def _create_faces(self):
        """Draw the complete object from scratch. Only runs once upon creating the instance"""
        self._polygon_ids = []
        for face in self._faces:
            draw_points = []
            for point_index in face:
                draw_points.extend(self._projected_points[point_index])
            polygon = self._canvas.create_polygon(
                draw_points, outline=self._line_colour, fill="")
            self._polygon_ids.append(polygon)
            
    def _update_faces(self):
        """Update polygon coordinates to refresh the canvas"""
        for polygon_id, face in zip(self._polygon_ids, self._faces):
            draw_points = []
            for point_index in face:
                draw_points.extend(self._projected_points[point_index])
            self._canvas.coords(
                polygon_id,
                *draw_points
            )

    def move(self, direction, amount):
        """Two-dimentional translation of the object"""
        directions = {
            '<Up>'   : (1, -1),
            '<Down>' : (1, 1),
            '<Left>' : (0, -1),
            '<Right>': (0, 1)
        }
        index, sign = directions[direction]
        self._position[index] += sign * amount
        self._changed = True

    def reset_rotation(self):
        """Reset rotation to (0, 0, 0)"""
        self._rotation = np.array([0]*3)

    @classmethod
    def from_file(cls, canvas, filename='', *args, **kwargs):
        if not filename:
            filename = filedialog.askopenfilename(defaultextension='.obj',
                    filetypes=(('OBJ Files', '*.obj'),
                               ('All Files', '*.*')))
            extension = filename[filename.rfind('.'):]
            if extension != '.obj':
                message = f'Invalid format {extension} - only .obj files allowed.'
                messagebox.showinfo(message=message, title="ERROR")
        return cls(filename, canvas, *args, **kwargs)
