import numpy as np

def add_ground_features_to_dataframe(df):
    df['grnd-ry'] = df['right-y'] - df['nose-y']

    # TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
    pass

def std_by_speaker(df):

    # TODO Return a dataframe named with standard deviations grouped by speaker
    pass

def add_normalized_features_to_dataframe(df, df_means, df_std):
    # TODO add features for normalized by speaker values of left, right, x, y
    # Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
    # using Z-score scaling (X-Xmean)/Xstd
    pass

def add_polar_features_to_dataframe(df):
    # TODO add features for polar coordinate values where the nose is the origin
    # Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
    # Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
    pass


def add_delta_features_to_dataframe(df):
    # TODO add features for polar coordinate values where the nose is the origin
    # Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
    # Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
    pass
