from monai.transforms import Resized, Spacingd, SpatialPadd, ScaleIntensityRanged, ScaleIntensityRangePercentilesd, HistogramNormalized, Orientationd, NormalizeIntensityd

transform_lib = {"HistogramNormalized": HistogramNormalized, "NormalizeIntensityd": NormalizeIntensityd, "Orientationd": Orientationd, "Spacingd": Spacingd, 
                "ScaleIntensityRanged": ScaleIntensityRanged, "ScaleIntensityRangePercentilesd": ScaleIntensityRangePercentilesd, 
                "SpatialPadd": SpatialPadd, "Resized": Resized}


