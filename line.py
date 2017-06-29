import numpy as np
import collections

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,
                 x_m_per_pix=1.,
                 y_m_per_pix=1.,
                 alpha=0.5,
                 n=5):
        # X meters per pixel
        self.x_m_per_pix = x_m_per_pix
        # Y meters per pixel
        self.y_m_per_pix = y_m_per_pix
        # Exponential averaging parameter, alpha
        self.alpha = alpha
        self.n = n
        # Curvertaure tolerance in m
        self.curvature_tolerance = 10.
        # Gradient tolerance in m/m
        self.gradient_tolerance = 1.
        # Separation tolerance
        self.max_separation = 750.
        self.min_separation = 250.
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0., 0., 0.])  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([None])] 
        self.last_n_fits = collections.deque(maxlen=self.n)
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #gradient closest to car
        self.gradient = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update(self,
               new_fit,
               other_line,
               allx,
               ally,
               image_shape,
               frame_number):
        self.allx = allx
        self.ally = ally

        ploty = np.linspace(0, image_shape[0]-1, image_shape[0])
        y_eval = np.max(ploty) # Evaluate line properties closest to vehicle

        if new_fit is not None:
            self.current_fit = new_fit
            new_radius_of_curvature = self.calculate_curvature(y_eval,
                                                               new_fit)

            other_curvature = self.calculate_curvature(y_eval,
                                                       other_line)

            self.gradient = self.calculate_gradient(y_eval,
                                                    new_fit)
            other_gradient = self.calculate_gradient(y_eval,
                                                     other_line)

            self.line_base_pos = self.get_line_base_pos(y_eval,
                                                        new_fit)
            other_line_base_pos = self.get_line_base_pos(y_eval,
                                                         other_line)

            # Sanity Checks
            line_is_sane = True
            # Check for similar radius
            if np.abs(new_radius_of_curvature - other_curvature) / new_radius_of_curvature > self.curvature_tolerance:
                line_is_sane = False
                # print("Not curvature {0} || {1}".format(self.radius_of_curvature,
                #                                            other_curvature))
            # Check if lines are parallel
            if np.abs(self.gradient - other_gradient) / self.gradient > self.gradient_tolerance or \
               np.sign(self.gradient) != np.sign(other_gradient):
                line_is_sane = False
                # print("Not parallel {0} || {1}".format(self.gradient,
                #                                            other_gradient))
            # Check lines are not too far apart
            if np.abs(self.line_base_pos - other_line_base_pos) > self.max_separation or \
               np.abs(self.line_base_pos - other_line_base_pos) < self.min_separation:
                line_is_sane = False
                # print("Not close {0} || {1}".format(self.line_base_pos,
                #                                            other_line_base_pos))
                # print("Sep {0}".format(np.abs(self.line_base_pos - other_line_base_pos)))
            
                        

            if line_is_sane:
                self.detected = True
                # self.best_fit = self.alpha * new_fit + (1.-self.alpha)*self.best_fit
                self.last_n_fits.append(new_fit)
                self.best_fit = np.mean(self.last_n_fits, 0)
                # Only update radius every so many frames
                if frame_number % 5 == 0:
                    # Used averaged curves to calculate radius
                    if self.radius_of_curvature is None:
                        self.radius_of_curvature = self.calculate_curvature(y_eval)
                    else:
                        self.radius_of_curvature = self.alpha * self.calculate_curvature(y_eval) + (1.-self.alpha)*self.radius_of_curvature
        else:
            self.detected = False

        return

    def calculate_gradient(self,
                           y_eval,
                           line_fit=None):

        if line_fit is None:
            line_fit = self.best_fit

        try:
            # Calculate the gradient of the curve closest to car
            curve_gradient = 2 * line_fit[0] * y_eval + line_fit[1]
        except:
            return 0.

        return curve_gradient


    def calculate_curvature(self,
                            y_eval,
                            line_fit=None):

        if line_fit is None:
            line_fit = self.best_fit

        # Calculate the new radii of curvature
        try:
            curve_radius = ((1 + (2*line_fit[0]*self.x_m_per_pix/self.y_m_per_pix**2*y_eval*self.y_m_per_pix \
                               + line_fit[1]*self.x_m_per_pix/self.y_m_per_pix)**2)**1.5) \
                             / np.abs(2*line_fit[0]*self.x_m_per_pix/self.y_m_per_pix**2)
        except:
            return 0.
        # if self.radius_of_curvature is None:
        #     self.radius_of_curvature = curve_radius
        # else:
        #     self.radius_of_curvature = self.alpha*curve_radius + (1 - self.alpha)*self.radius_of_curvature

        return curve_radius

    def get_line_base_pos(self,
                          y_eval,
                          line_fit=None):
        if line_fit is None:
            line_fit = self.best_fit

        try:
            base_pos = line_fit[0]*y_eval**2 + line_fit[1]*y_eval + line_fit[2]
        except:
            return 0.

        return base_pos



