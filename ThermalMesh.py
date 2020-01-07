#!/usr/bin/env python3
""" ThermalMesh.py - Finite difference thermal diffusivity model.

This module implements a rectangular mesh for simulating the 
temperature evolution versus time of a 2D plate. This is also 
called a "thermal diffusivity" simulation.

    Typical usage example:
    # Create mesh, set scale. `MFN` is the Fourier Number
    # (Sets how fast heat transfers in the plate)
    myPlate = ThermalMesh(nX=40, nY=30, dx=1, dt=1, MFN = 0.1)
    
    # template set mode of each node
    myPlate.setTemplate(template) 
    
    # Initialize plate to a constant temperature
    myPlate.setInitT(self, initT) 
    
    # `beta` is the rate of Newton's law of cooling to the ambient
    myPlate.setBeta(beta)
    # Set ambient temperature the plate cools (or heats) toward
    myPlate.setTAmb(TAmb)
    
    # Set adiabatic boundary conditions
    myPlate.setBorderAdiabatic()
    
    # (Optional) Set constant heat input sources on plate
    myPlate.setCQSources(CQarray)
    
    # (Optional) Set constant temperature nodes
    # Can include temperare profile on boundary
    myPlate.setConstT(CTarray)
    
    # Do one time step iteration
    myPlate.iterate()

    # Save plate temperature as a PGM image
    myPlate.savePGM(fileName)
    
    # Save plate temperature as a PNG image
    myPlate.savePNG(fileName)
"""

"""    
## Versions
- 19a 
  - Totally redefine approach. The main class is ThermalMesh with the temperature 
    stored in a 2D `numpy` array.
  - `savePGM` to write an image file
  - `setBorderAdiabatic` to set no heat flow out of the side
  - Implemented _Newton's Law of Cooling_
  - `iterate` using equation below
  
- 19b
  - Add heat input to certain nodes
  - Add constant Temperature nodes
  
- 19c
  - Clean up code and documentation
  - Change image & animation code
 
- 20a
  - Remove `setZeros`
  - Add `setInitT` to initialize the mes temperature
  - Add explicity _template_ to class. A template describes what a node is
    or does. A template must be set before calculation can be done Examples are:
    - .: a normal temperature evolving node
    - Q: constant heat input to a node
    - T: constant termperature node (Dirchlet condition when on boundary)
    - P: a periodic boundary node (Not implemented, requires resizing nodes)
    - A: an adiabatic border node (Neumann condition)
  - iterate modifies self instead of creating a new object
"""

import numpy as np
from copy import copy, deepcopy
from PIL import Image

class THERMAL_MESH_ERROR(Exception):
    pass

class ThermalMesh(object):
    """ Implement a mesh for solving heat transfer problems. 
    
    TODO:
        - Implement periodic boundary conditions.
    """

    _BORDER_TYPES = frozenset(['Adiabatic', 'ConstT', 'Mixed'])

    def __init__(self, nX=4, nY=3, dx=1, dt=1, MFN = 0.1, border='Adiabatic'):
        """ Creator initializes ThermalMesh """
        self.nX = int(nX)
        self.nY = int(nY)
        self.dx = dx
        self.dt = dt
        self.setMFN(MFN)
        self.nCols = self.nX+2
        self.nRows = self.nY+2
        self.nodes = np.zeros((self.nRows, self.nCols), dtype=float)
        self.shape = self.nodes.shape
        if border in self._BORDER_TYPES:
            self.border = border
        else:
            raise THERMAL_MESH_ERROR(f"Border type error, {border} not implemented")
            
    def __str__(self):
        """ Print information about mesh. """
        strn = f"Mesh({self.nX},{self.nY}): \n"
        bNewLine = False
        if hasattr(self,'MFN'):
            bNewLine = True
            strn += f"MFN = {self.MFN}"
        if hasattr(self,'beta'):
            if bNewLine:
                strn += ", "
            else:
                bNewLine = True
            strn += f"beta = {self.beta}"
        if hasattr(self,'TAmb'):
            if bNewLine:
                strn += ", "
            else:
                bNewLine = True
            strn += f"TAmb = {self.TAmb}"
        if bNewLine:
            strn += '\n'
        strn += str(self.nodes)
        return strn
    
    def setInitT(self, initT):
        """ Initializes temperatures of all modes. 
        
            Can be used to set all nodes to one temperature or if an
            array is supplied, individually set each node temperature.
        """
        if not np.isscalar(initT) and initT.shape != (self.nY, self.nX):
            raise THERMAL_MESH_ERROR(\
                f"setInitT shape mismatch {initT.shape} not {(self.nY-2, self.nX-2)}")
        self.nodes[1:self.nY+1, 1:self.nX+1] = initT
        self.setBordersAdiabatic()
        
    def setTemplate(self, template):
        if len(template) != self.nY+2 or \
            len(template[0]) != self.nX+2:
            raise THERMAL_MESH_ERROR(\
                f"Template shape, {template}, mismatch to {self.nX+2}, {self.nY+2}")
            
        # Set border flags
        self.template = template
        self.topAdiabatic = False
        if self.template[0][1:-1] == self.nX*"A":
            self.topAdiabatic = True
        self.bottomAdiabatic = False
        if self.template[-1][1:-1] == self.nX*"A":
            self.bottomAdiabatic = True

        self.leftAdiabatic = True
        for iR in range(1,self.nY-1):
            if self.template[iR][0] != 'A':
                self.leftAdiabatic = False

        self.rightAdiabatic = True
        for iR in range(1,self.nY-1):
            if self.template[iR][-1] != 'A':
                self.rightAdiabatic = False
        print("setTemplate: ", self.topAdiabatic, self.bottomAdiabatic, \
              self.leftAdiabatic, self.rightAdiabatic)

        
    def setT(self, ix, iy, T):
        """ Set the temperature of a node. """
        self.nodes[iy+1, ix+1] = T
    
    def setMFN(self, MFN=0.1):
        """ Set MFN, the mesh Fourier Number. """
        self.MFN = MFN
    
    def setBeta(self, beta):
        """ Set the coefficient of cooling. """
        self.beta = beta
    
    def setTAmb(self, TAmb):
        """ Set the ambient temperature. """
        self.TAmb = TAmb
    
    def setBordersAdiabatic(self):
        """ Set all of the border to adiabatic. """
        if self.topAdiabatic:
            # print("top", end=" ")
            self.nodes[0,:] = self.nodes[1,:]
        if self.bottomAdiabatic:
            # print("bottom", end=" ")
            self.nodes[self.nY+1,:] = self.nodes[self.nY,:]
        if self.leftAdiabatic:
            # print("left", end=" ")
            self.nodes[:,0] = self.nodes[:,1]
        if self.rightAdiabatic:
            # print("right")
            self.nodes[:,self.nX+1] = self.nodes[:,self.nX]
        
    def setCQSources(self, array):
        if self.nodes.shape != array.shape:
            raise THERMAL_MESH_ERROR( \
              f"setCQSources: {self.node.shape} != {array.shape}")
        self.CQSources = copy(array)
        self.whereCQ = np.where(self.CQSources > 0)
# print(f"CQSources min = {np.min(self.CQSources)}, max = {np.max(self.CQSources)}")
        
    def setConstT(self, array):
        # Set the constant temperature array
        if self.nodes.shape != array.shape:
            raise THERMAL_MESH_ERROR( \
              f"setConstT: {self.nodes.shape} != {array.shape}")
        self.constT = copy(array)
        self.whereConstT = np.where(self.constT > 0)
        self._setConstT()
        
    def _setConstT(self):
        self.nodes[self.whereConstT] = self.constT[self.whereConstT]
        
    def _xRoll(self, nXRoll):
        return np.roll(self.nodes,nXRoll,1)
        
    def _yRoll(self, nYRoll):
        return np.roll(self.nodes,nYRoll,0)
    
    def iterate(self):
        newNodes = self.nodes + \
            (self.MFN * self.dt / self.dx**2) * \
            (self._xRoll(1) + self._xRoll(-1) + \
            self._yRoll(1) + self._yRoll(-1) - 4 * self.nodes) - \
            self.beta * self.dt * (self.nodes - self.TAmb)
        
        # If constant heat sources, increase T by dq/dt & Dt
        if hasattr(self,'CQSources'):
            newNodes[self.whereCQ] += self.dt * self.CQSources[self.whereCQ]
            
        self.nodes = newNodes

        # If constant T pixels, copy T from constT array to new nodes
        if hasattr(self,'constT'):
            self._setConstT()
        
        self.setBordersAdiabatic()
        return
        
    def savePGM(self, fileName):
        if fileName.lower().find('.pgm') != (len(fileName)-4):
            raise THERMAL_MESH_ERROR(f"savePGM: file name {fileName} must end in '.pgm'")
        with open(fileName, 'w') as fpPGM:
            fpPGM.write("P2\n")
            fpPGM.write(f"{int(self.nX)} {int(self.nY)}\n")
            
            imgData = self.nodes[1:1+self.nY,1:1+self.nX]
            maxVal = int(np.max(imgData)+0.5)
            while maxVal > 65536:
                print("savePGM: dividing by 10")
                imgData /= 10
                maxVal = int(np.max(imgData)+0.5)
            if (maxVal <= 0) or (maxVal > 65536):
                raise THERMAL_MESH_ERROR(f"savePGM, maxVal = {maxVal}")
            fpPGM.write(f"{int(maxVal+0.5)}\n")
            
            maxVal = int(np.max(imgData))
            minVal = int(np.min(imgData))
            if minVal < 0:
                raise THERMAL_MESH_ERROR(f"savePGM, minVal = {minVal} < 0")
            for iy in range(self.nY):
                for ix in range(self.nX):
                    fpPGM.write(f"{int(imgData[iy,ix]+0.5)} ")
                fpPGM.write("\n")
        # print(f"Wrote {fileName}")
    
    def savePNG(self, fileName):
        """ Save nodes as a PNG image normalized from 0 to 255
        """
        if fileName.lower().find('.png') != (len(fileName)-4):
            raise THERMAL_MESH_ERROR(f"savePNG: file name {fileName} must end in '.png'")
        saveImg = self.nodes[1:1+self.nY, 1:1+self.nX]
        minV = np.min(saveImg)
        maxV = np.max(saveImg)
        if maxV == minV:
            saveImg = np.zeros_like(saveImg, dtype='uint8')
        else:
            saveImg = (255 * (saveImg - minV)/(maxV - minV)).astype('uint8')
            
        im = Image.fromarray(saveImg)
        im.save(fileName)
            
def main():
    # Defaults to nX=4, nY=3, dx=1, dt=1, MFN = 0.1, border='Adiabatic'
    myMesh0 = ThermalMesh()
    template = [
        "AAAAAA",
        "A.T..A",
        "A...QA",
        "ATTTTA",
        "ATTTTA",
               ]
    print("template:")
    for i in range(len(template)):
        print(template[i])
    myMesh0.setTemplate(template)
    print(myMesh0)
    initT = 30.0
    myMesh0.setInitT(initT) 
    myMesh0.setBordersAdiabatic()
    print(myMesh0)

    beta = 0.2
    myMesh0.setBeta(beta)
    
    # Initialize 
    ambientT = 25.0
    myMesh0.setTAmb(ambientT)
    
    # Set initial temperature with an array of t's
    val = ambientT
    initT = np.zeros(shape=(3,4), dtype=float)
    for ix in range(4):
        for iy in range(3):
            val += 1
            initT[iy, ix] = val
    print(f"initT = \n{initT}")
            
    myMesh0.setInitT(initT)
    print(myMesh0)

    # Set heat sources
    CQarray = np.zeros_like(myMesh0.nodes)
    CQarray[2,4] = 0.2
    myMesh0.setCQSources(CQarray)
    print(f"CQSources = \n{myMesh0.CQSources}")
         
    # Set constant temperature nodes
    CTarray = np.zeros_like(myMesh0.nodes)
    CTarray[3,1:-1] = np.array([40.0,41,42,43])
    CTarray[1,2] = 45
    myMesh0.setConstT(CTarray)
    print(f"constT = \n{myMesh0.constT}")

    print(myMesh0)
    
    print("S T A R T   I T E R A T I O N")
    myMesh0.iterate()
    print("After iteration:\n", myMesh0)
    myMesh0._setConstT()
    print("After _setConstT:\n", myMesh0)
    myMesh0.setBordersAdiabatic()
    print("After setBordersAdiabatic:\n", myMesh0)
    myMesh0.savePGM("tesTAmb.pgm")
    myMesh0.savePNG("tesTAmb.png")
    print("Iterated")

if __name__ == '__main__':
    main()
    
# End