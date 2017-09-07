import vtk
import vtk.util.colors as vtk_clr


def save_nii2avi(niipath='test_0.nii.gz', Save_file_name="moive.mp4", Time_Loop=20, Avi_rate=5, Angle=20):
    # read a NIFTI file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(niipath)
    reader.TimeAsVectorOn()
    reader.Update()
    def getActor(b1,b2,color):
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputConnection(reader.GetOutputPort())
        threshold.ThresholdBetween(b1, b2)
        threshold.ReplaceInOn()
        threshold.SetInValue(0)  # set all values below 400 to 0
        threshold.ReplaceOutOn()
        threshold.SetOutValue(1)  # set all values above 400 to 1
        threshold.Update()
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputConnection(threshold.GetOutputPort())
        dmc.GenerateValues(1, 1, 1)
        dmc.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(dmc.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        return actor

    actor_1 = getActor(200,206,vtk_clr.banana)
    actor_2 = getActor(410,430,vtk_clr.blue_light)
    actor_3 = getActor(490,510,vtk_clr.cadmium_lemon)
    actor_4 = getActor(540,560,vtk_clr.chrome_oxide_green)
    actor_5 = getActor(590,610,vtk_clr.rose_madder)
    actor_6 = getActor(810,830,vtk_clr.green_dark)
    actor_7 = getActor(840,860,vtk_clr.purple)
    #Renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    #RenderWindow
    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(renderer)
    #assemble all part
    assembly = vtk.vtkAssembly()
    assembly.AddPart(actor_1)
    assembly.AddPart(actor_2)
    assembly.AddPart(actor_3)
    assembly.AddPart(actor_4)
    assembly.AddPart(actor_5)
    assembly.AddPart(actor_6)
    assembly.AddPart(actor_7)
    assembly.SetOrigin(0,0,0)
    #Add outline assenble actor
    renderer.AddActor(assembly)
    renwin.SetSize(600,600)
    # interactor = vtk.vtkRenderWindowInteractor()
    # interactor.SetRenderWindow(renwin)
    # interactor.Initialize()
    renwin.Render()
    #convert console to movie
    imageFilter = vtk.vtkWindowToImageFilter()
    imageFilter.SetInput(renwin)
    moviewriter = vtk.vtkOggTheoraWriter()
    moviewriter.SetInputConnection(imageFilter.GetOutputPort())
    moviewriter.SetFileName(Save_file_name)
    moviewriter.Start()
    moviewriter.SetRate(Avi_rate)
    for i in range(Time_Loop):
        renderer.GetActiveCamera().Azimuth(Angle)
        imageFilter.Modified()
        moviewriter.Write()
    moviewriter.End()

