import numpy as np
from numpy import math
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['image.cmap'] = 'jet'
import bornagain as ba
from bornagain import deg, nm


first_run = True
batch_sim = True

#  In case the full fitting procedure is used instead of single batch simulations,
#  the initial parameters must be chosen in the global scope:

#init_height =
#init_radius =
#init_height_flattening_ratio =
#init_lattice_length =
#init_damping_length =
#init_disorder_parameter =
#init_beam_intensity =


class PlotObserver():  #defines image canvas displayed on the screen during running
    def __init__(self):
        self.fig = plt.figure(figsize=(10.25, 7.69))
        self.fig.canvas.draw()

    def __call__(self, fit_objective):
        self.plot_update(fit_objective)

    @staticmethod
    def make_subplot(nplot):
        plt.subplot(2, 2, nplot)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

    def plot_update(self, fit_objective):
        self.fig.clf()

        real_data = fit_objective.experimentalData()
        sim_data = fit_objective.simulationResult()
        diff = fit_objective.relativeDifference()

        self.make_subplot(1)

        zmin = 0.1
        zmax = 100

        ba.plot_colormap(real_data, title="Experimental data", zmin=zmin, zmax=zmax,
                         units=ba.AxesUnits.QSPACE, xlabel=None, ylabel=None,
                         zlabel='', aspect=None)

        self.make_subplot(2)
        ba.plot_colormap(sim_data, title="Simulation result", zmin=zmin, zmax=zmax,
                         units=ba.AxesUnits.QSPACE, xlabel=None, ylabel=None,
                         zlabel='', aspect=None)

        self.make_subplot(3)
        ba.plot_colormap(diff, title="Relative difference", zmin=0.001, zmax=10.0,
                         units=ba.AxesUnits.QSPACE, xlabel=None, ylabel=None,
                         zlabel='', aspect=None)

        self.make_subplot(4)

        plt.title('Parameters')
        plt.axis('off')

        iteration_info = fit_objective.iterationInfo()

        plt.text(0.01, 0.85, "Iterations  " + '{:d}'.
                 format(iteration_info.iterationCount()))
        plt.text(0.01, 0.75, "Chi2       " + '{:8.4f}'.format(iteration_info.chi2()))
        index = 0
        params = iteration_info.parameterMap()
        for key in params:
            plt.text(0.01, 0.55 - index * 0.1, '{:30.30s}: {:6.3f}'.format(key, params[key]))
            index = index + 1

        self.fig.tight_layout()
        plt.pause(0.03)


def get_sample(params):  #  function which constructs the sample

    radius = params["radius"]
    height = params["height"]
    height_flattening = (params["height"]/params["radius"])*params["height_flattening_ratio"]
    lattice_length = params["lattice_length"]
    damping_length = params["damping_length"]
    disorder_parameter = params["disorder_parameter"]

    # Defining materials:
    m_air = ba.MaterialBySLD("Air", 0.0, 0.0)
    m_Cu = ba.MaterialBySLD("Cu", 6.9508e-05, 5.142e-06)
    m_PEO = ba.MaterialBySLD("PEO", 1.0463e-05, 7.9515e-09)
    #m_SiO2 = ba.MaterialBySLD("SiO2", 1.8749e-05, 9.3923e-08)
    #m_Si = ba.MaterialBySLD("Si", 1.9893e-05, 1.795e-07)

    # Defining particles:
    tr_spheroid_ff = ba.FormFactorTruncatedSpheroid(radius, height, height_flattening)
    particle = ba.Particle(m_Cu, tr_spheroid_ff)

    # Defining interference function:
    interference_f = ba.InterferenceFunctionRadialParaCrystal(lattice_length, damping_length)
    omega = disorder_parameter * lattice_length
    pdf = ba.FTDistribution1DGauss(omega)
    interference_f.setProbabilityDistribution(pdf)

    # Defining layers:
    layout = ba.ParticleLayout()
    layout.addParticle(particle, 1.0)
    layout.setInterferenceFunction(interference_f)
    layout.setTotalParticleSurfaceDensity(2/(math.sqrt(3)*lattice_length*lattice_length))

    air_layer = ba.Layer(m_air)
    air_layer.setNumberOfSlices(15)
    air_layer.addLayout(layout)

    PEO_layer = ba.Layer(m_PEO)
    #PEO_layer = ba.Layer(m_PEO, 90*nm)
    #Si_layer = ba.Layer(m_Si)

    #roughness = ba.LayerRoughness()
    #roughness.setSigma(10.0*nm)
    #roughness.setHurstParameter(1.0)
    #roughness.setLatteralCorrLength(1.0e-5*nm)

    # Assembling the multilayer:
    multi_layer = ba.MultiLayer()
    multi_layer.addLayer(air_layer)
    multi_layer.addLayer(PEO_layer)

    #multi_layer.addLayerWithTopRoughness(PEO_layer, roughness)
    #multi_layer.addLayer(Si_layer)

    return multi_layer

def get_simulation(params):  # function which defines the GISAXS simulation

    simulation = ba.GISASSimulation()

    detector = ba.RectangularDetector(981, 168.732, 1043, 179.396)  # creates the Pilatus detector
    detector.setPerpendicularToDirectBeam(2385.807, 89.522, 21.83)  # adds the SDD and DB position to the detector
    simulation.setDetector(detector)  # adds the detector to the simulation

    simulation.setBeamParameters(0.095373 * nm, 0.378 * deg, 0.0 * deg)  # adds the beam parameters to the simulation
    beam_intensity = params["beam_intensity"]
    simulation.setBeamIntensity(beam_intensity)
    #simulation.setBackground(ba.PoissonNoiseBackground())

    simulation.setSample(get_sample(params))  #  adds the sample to the simulation


    #print(simulation.treeToString())
    print(simulation.parametersToString())

    simulation.getOptions().setIncludeSpecular(False)
    simulation.getOptions().setUseAvgMaterials(True)  # enables the graded interfaces formalism
    #simulation.getOptions().setMonteCarloIntegration(False, 50)

    simulation.setRegionOfInterest(85, 38, 164, 143)

    # Masks for full fitting procedure:

    #simulation.addMask(ba.Rectangle(83.696, 36.530, 84.951 , 106.469), True)
    #simulation.addMask(ba.Rectangle(3.732, 70.213, 154.232, 73.011), True)
    #simulation.addMask(ba.Ellipse(59.501, 48.764, 0.239, 0.181), True)
    #simulation.addMask(ba.Ellipse(79.088, 54.814, 3.132, 3.338), True)
    #simulation.addMask(ba.Ellipse(90.57, 105.700, 0.167, 0.191), True)
    #simulation.addMask(ba.Polygon([76.29, 76.88, 7.59, 5.51],[56.26, 57.17, 106.48, 106.53]), True)


    # Size distribution:

    #n_samples = 20
    #R = params["radius"]
    #R_stdev = 0.2*R
    #R_min = 0.1*R
    #R_max = 2*R
    #sigma_par = 2.0*nm
    #radius_distr = ba.DistributionGaussian(R, R_stdev)
    #simulation.addParameterDistribution("*/Radius", radius_distr, n_samples, sigma_par, ba.RealLimits.limited(R_min, R_max))

    simulation.setTerminalProgressMonitor()
    return simulation

def load_real_data(i_img):  #  function to read experimental files
    # data_folder = "C:/Users/ge56lij/PycharmProjects/ProjectSample#32/InputData/"
    data_folder = "C:/Users/Valentin1/PycharmProjects/ProjectSample#32/InputData/"
    if i_img < 10:
        filename = data_folder + "cu_sjsb1_32_00012_SUMResult_0000" + repr(i_img) + "_0000" + repr(i_img) + ".tif"
    elif i_img < 100:
        filename = data_folder + "cu_sjsb1_32_00012_SUMResult_000" + repr(i_img) + "_000" + repr(i_img) + ".tif"
    elif i_img < 1000:
        filename = data_folder + "cu_sjsb1_32_00012_SUMResult_00" + repr(i_img) + "_00" + repr(i_img) + ".tif"
    else:
        filename = data_folder + "cu_sjsb1_32_00012_SUMResult_0" + repr(i_img) + "_0" + repr(i_img) + ".tif"
    print(filename)

    return ba.IntensityDataIOFactory.readIntensityData(filename).array() + 1.0e-10

def run_fitting(i_img):  # for this thesis just batch simulations, no fitting.
                         # passing the simulation through the fitting mechanism
                         # still useful for processing experimental and
                         # simulation patterns simultaneously

    global first_run
    global batch_sim

    global init_radius
    global init_height
    global init_height_flattening_ratio
    global init_lattice_length
    global init_damping_length
    global init_disorder_parameter
    global init_beam_intensity

    if batch_sim==True:  # setting the simulation parameters
        d_eff = 11.6 * (0.1 * (i_img - 11)) / 150
        init_lattice_length = 2 * np.pi / (0.628 + 1.499 * math.exp(-d_eff/3.31) )
        init_height = 1.519 + 1.367*d_eff - 0.038*(d_eff**2)
        init_radius = init_height/3.1
        init_height_flattening_ratio = 0.66
        init_disorder_parameter = 0.25
        init_damping_length = 30 * nm
        init_beam_intensity = 4.15e+09

    real_data = load_real_data(i_img)

    fit_objective = ba.FitObjective()
    fit_objective.addSimulationAndData(get_simulation, real_data)

    fit_objective.initPrint(1)
    fit_objective.initPlot(1, PlotObserver())  #plotting frequency on screen (increase if OpenGL error)

    params = ba.Parameters()  #parameter limits only relevant for full fitting, otherwise ignored

    min_height = 0.1
    max_height = 20
    min_lattice_length = 0.1
    max_lattice_length = 20
    min_radius = 0.1
    max_radius = 20
    min_damping_length = 10
    max_damping_length = 1000
    min_disorder_parameter = 0.1
    max_disorder_parameter = 0.5
    min_beam_intensity = 1e+8
    max_beam_intensity = 1e+12

    params.add("radius", value=init_radius, min=min_radius, max=max_radius, step=0.1*nm, vary=True)
    params.add("height", value=init_height, min=min_height, max=max_height, step=0.1*nm, vary=True)
    params.add("height_flattening_ratio", value=init_height_flattening_ratio, min=0.55, max=0.75, step=0.01, vary=True)
    params.add("lattice_length", value=init_lattice_length, min=min_lattice_length, max=max_lattice_length, step=0.1*nm,
               vary=True)
    params.add("damping_length", value=init_damping_length, min=min_damping_length, max=max_damping_length, step=10*nm,
               vary=True)
    params.add("disorder_parameter", value=init_disorder_parameter, min=min_disorder_parameter,
               max=max_disorder_parameter, step=0.01, vary=True)
    params.add("beam_intensity",init_beam_intensity, min=min_beam_intensity, max=max_beam_intensity, vary=False)


    minimizer = ba.Minimizer()
    minimizer.setMinimizer("Test")  # normal simulation (no fitting)
    #minimizer.setMinimizer("Minuit2", "Migrad", "MaxFunctionCalls=0;Strategy=2;")  # minimizer for fitting

    result = minimizer.minimize(fit_objective.evaluate, params)  # runs the simulation
    fit_objective.finalize(result)

    fig_name = output_folder + 'fitting_img_#32_' + repr(i_img) + '.png'
    plt.savefig(fig_name, dpi=500)
    plt.close(1)

    print("Fitting completed.")
    print("chi2:", result.minValue())
    for fitPar in result.parameters():
        print(fitPar.name(), fitPar.value, fitPar.error)

    experimental_data = fit_objective.experimentalData()
    simulation_data = fit_objective.simulationResult()
    difference = fit_objective.relativeDifference()

    zmin = 0.1  #  range of the colormap
    zmax = 100

    mpl.rcParams['image.cmap'] = 'jet'  #  saves figures in jet colormap

    plt.figure(2,figsize=(8, 6))
    ba.plot_colormap(experimental_data, title="Experimental data", zmin=zmin, zmax=zmax,
                     units=ba.AxesUnits.QSPACE, xlabel=None, ylabel=None, zlabel='', aspect=None)
    fig_name = output_folder + 'exp_data_#32_' + repr(i_img) + '_jet' + '.png'
    plt.savefig(fig_name, dpi=500)
    plt.close(2)

    plt.figure(3,figsize=(8, 6))
    ba.plot_colormap(simulation_data, title="Simulation result", zmin=zmin, zmax=zmax,
                     units=ba.AxesUnits.QSPACE, xlabel=None, ylabel=None, zlabel='', aspect=None)
    fig_name = output_folder + 'sim_data_#32_' + repr(i_img) + '_jet' + '.png'
    plt.savefig(fig_name, dpi=500)
    plt.close(3)

    mpl.rcParams['image.cmap'] = 'nipy_spectral'  #saves figures in spectral colormap

    plt.figure(4, figsize=(8, 6))
    ba.plot_colormap(experimental_data, title="Experimental data", zmin=zmin, zmax=zmax,
                     units=ba.AxesUnits.QSPACE, xlabel=None, ylabel=None, zlabel='', aspect=None)
    fig_name = output_folder + 'exp_data_#32_' + repr(i_img) + '_spectrum' + '.png'
    plt.savefig(fig_name, dpi=500)
    plt.close(4)

    plt.figure(5, figsize=(8, 6))
    ba.plot_colormap(simulation_data, title="Simulation result", zmin=zmin, zmax=zmax,
                     units=ba.AxesUnits.QSPACE, xlabel=None, ylabel=None, zlabel='', aspect=None)
    fig_name = output_folder + 'sim_data_#32_' + repr(i_img) + '_spectrum' + '.png'
    plt.savefig(fig_name, dpi=500)
    plt.close(5)

    #  Saves extra information for full fitting procedure:

    # txt_name = output_folder + 'fitting_result_#32_' + repr(i_img) + '.txt'
    # fout = open(txt_name, 'w')
    #
    # iteration_info = fit_objective.iterationInfo()
    # fit_params = iteration_info.parameterMap()
    #
    # fout.write("# Iterations:  " + '{:d}\n'.format(iteration_info.iterationCount()))
    # fout.write("Chi2       " + '{:8.4f}\n'.format(iteration_info.chi2()))
    #
    # for key in fit_params:
    #     fout.write('{:30.30s}: {:6.3f}\n'.format(key, fit_params[key]))
    #
    # table_name = output_folder + 'fitting_table_#32.txt'
    # table_output = open(table_name, 'a')
    #
    # if first_run == True:
    #     first_run = False
    #     table_output.write('{:30.30s}'.format("Image #"))
    #     table_output.write('{:30.30s}'.format("Iterations #"))
    #     table_output.write('{:30.30s}'.format("Chi2"))
    #     for key in fit_params:
    #         table_output.write('{:30.30s}'.format(key))
    #     table_output.write('{:30.30s}'.format("e_spheroid"))
    #     table_output.write('{:30.30s}'.format("height_flattening"))
    #     table_output.write('\n')
    #
    # iteration_info = fit_objective.iterationInfo()
    # fit_params = iteration_info.parameterMap()
    # table_output.write('{:<30d}'.format(i_img))
    # table_output.write('{:<30d}'.format(iteration_info.iterationCount()))
    # table_output.write('{:<30.5f}'.format(iteration_info.chi2()))
    # for key in fit_params:
    #     table_output.write('{:<30.9f}'.format(fit_params[key]))
    # table_output.write('{:<30.9f}'.format(fit_params["height"]/fit_params["radius"]))
    # table_output.write('{:<30.9f}'.format((fit_params["height"]/fit_params["radius"])*
    #                                       fit_params["height_flattening_ratio"]))
    # table_output.write('\n')
    #
    # np.savetxt(output_folder + 'simulation_result_#32_' + repr(i_img) + '.txt', fit_objective.simulationResult().array())


if __name__ == '__main__':
    # output_folder = "C:/Users/ge56lij/PycharmProjects/ProjectSample#32/SimResults/"
    output_folder = "C:/Users/Valentin1/PycharmProjects/ProjectSample#32/SimResults/"
    for i_img in range(100, 1501, 100):  # frame range of the simulations
        try:
            run_fitting(i_img)
        except Exception:
            pass
