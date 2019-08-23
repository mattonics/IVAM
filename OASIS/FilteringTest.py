import scipy.ndimage.filters as filters
import CrossSectionalData

base_string = 'D:\MriData\Data'
excel_path = 'D:\oasis_cross-sectional.csv'


data = CrossSectionalData.CrossSectionalDataProvider(base_string,excel_path).retrieve_full_data_example()

data = filters.gaussian_filter(data,1.1)

CrossSectionalData.show_slices([data[:,:,50]])


