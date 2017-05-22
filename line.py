import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,
                 x_m_per_pix=1.,
                 y_m_per_pix=1.,
                 alpha=0.5):
        # X meters per pixel
        self.x_m_per_pix = x_m_per_pix
        # Y meters per pixel
        self.y_m_per_pix = y_m_per_pix
        # Exponential averaging parameter, alpha
        self.alpha = alpha
        # Curvertaure tolerance in m
        self.tolerance = 10
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
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
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
               image_shape):
        self.allx = allx
        self.ally = ally

        if new_fit is not None:
            self.current_fit = new_fit
            self.radius_of_curvature = self.calculate_curvature(image_shape,
                                                            new_fit)
            other_curvature = self.calculate_curvature(image_shape,
                                                       other_line)

            if np.abs(self.radius_of_curvature - other_curvature) / self.radius_of_curvature < self.tolerance:
                self.detected = True
                self.best_fit = self.alpha * new_fit + (1.-self.alpha)*self.best_fit
            else:
                print("Not parallel {0} || {1}".format(self.radius_of_curvature,
                                                       other_curvature))
        else:
            self.detected = False

        return

    def calculate_curvature(self,
                            image_shape,
                            line_fit=None):
        ploty = np.linspace(0, image_shape[0]-1, image_shape[0])
        y_eval = np.max(ploty)

        if line_fit is None:
            line_fit = self.best_fit

        # Calculate the new radii of curvature
        curve_radius = ((1 + (2*line_fit[0]*self.x_m_per_pix/self.y_m_per_pix**2*y_eval*self.y_m_per_pix \
                              + line_fit[1]*self.x_m_per_pix/self.y_m_per_pix)**2)**1.5) \
                           / np.abs(2*line_fit[0]*self.x_m_per_pix/self.y_m_per_pix**2)

        return curve_radius

    def set_y_plot_values(self,
                          binary_warped):
        self.ally = np.linspace(0,
                                binary_warped.shape[0]-1,
                                binary_warped.shape[0])

