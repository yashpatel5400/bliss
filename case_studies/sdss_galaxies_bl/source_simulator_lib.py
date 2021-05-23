import torch

import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

from bliss.models.decoder import Tiler
from bliss.models.encoder import get_mgrid

from which_device import device

#################
# Functions to set up and transform (aka rotate or stretch)
# coordinate system 
#################
def _get_mgrid(slen, normalize = True):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    
    mgrid = torch.Tensor(np.dstack((x, y))).to(device)
    
    if normalize: 
        mgrid = mgrid / offset
    
    return mgrid

def _get_rotation_matrix(theta): 
    
    batchsize = len(theta)
    
    sine_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    rotation = torch.dstack((cos_theta, -sine_theta,
                             sine_theta, cos_theta)).view(batchsize, 2, 2) 
    
    return rotation.to(device)

def _get_strech_matrix(ell): 
    
    batchsize = len(ell)
    
    stretch = torch.zeros(batchsize, 2, 2, device = device)
    
    stretch[:, 0, 0] = 1
    stretch[:, 1, 1] = ell
    
    return stretch

def _transform_mgrid_to_radius_grid(mgrid, theta, ell): 
    
    rotation = _get_rotation_matrix(theta)
    stretch = _get_strech_matrix(ell)
    
    rotated_basis = torch.einsum('nij, njk -> nik', rotation, stretch)
    precision = torch.einsum('nij, nkj -> nik', rotated_basis, rotated_basis)
    
    r2_grid = torch.einsum('xyi, nij, xyj -> nxy', 
                           mgrid, precision, mgrid)
    
    return r2_grid

#################
# galaxy profiles
#################
def render_gaussian_galaxies(r2_grid, half_light_radii): 
    
    assert r2_grid.shape[0] == len(half_light_radii)
    batchsize = len(half_light_radii)
    
    scale = half_light_radii.view(batchsize, 1, 1)**2 / np.log(2)
    
    return torch.exp(-r2_grid / scale)

def render_exponential_galaxies(r2_grid, half_light_radii): 
    
    assert r2_grid.shape[0] == len(half_light_radii)
    batchsize = len(half_light_radii)
    
    scale = half_light_radii.view(batchsize, 1, 1) / np.log(2)
    r_grid = torch.sqrt(r2_grid)
    
    return torch.exp(- r_grid / scale)

def render_devaucouleurs_galaxies(r2_grid, half_light_radii): 
    
    assert r2_grid.shape[0] == len(half_light_radii)
    batchsize = len(half_light_radii)

    scale = half_light_radii.view(batchsize, 1, 1)**(0.25) / np.log(2)
    r_grid = torch.sqrt(r2_grid)
    
    return torch.exp(-r_grid**0.25 / scale)

###############
# function to render centered galaxies
###############
def render_centered_galaxy(flux, theta, ell, r_dev, r_exp, p_dev, 
                           galaxy_mgrid): 
    
    # number of galaxies
    n_galaxies = len(flux)
    
    # radial coordinate system
    r2_grid = _transform_mgrid_to_radius_grid(galaxy_mgrid, 
                                              theta = theta,  
                                              ell = ell)
    
    # exponential profile
    exp_profile = render_exponential_galaxies(r2_grid, 
                                              half_light_radii = r_exp) 
    
    # dev. profile 
    dev_profile = render_exponential_galaxies(r2_grid, 
                                              half_light_radii = r_dev)
    
    # mixture weight and fluxes
    p_dev = p_dev.view(n_galaxies, 1, 1)
    flux = flux.view(n_galaxies, 1, 1)
    
    centered_galaxy = exp_profile * (1 - p_dev) + exp_profile * p_dev
    centered_galaxy *= flux
    
    return centered_galaxy.unsqueeze(1)

##################
# function to convolve w psf
##################

def _convolve_w_psf(images, psf): 
    
    # first dimension is number of bands
    assert len(psf.shape) == 3 
    
    # need to flip based on how the pytorch convolution works ... 
    _psf = psf.flip(-1).flip(1).unsqueeze(0)
    padding = int((psf.shape[-1] - 1) / 2)
    images = F.conv2d(images, 
                      _psf, 
                      stride = 1,
                      padding = padding)
    
    return images


###############
# class to render sources on a padded tiles 
###############

class SourceSimulator: 
    def __init__(self, 
                 psf,
                 tile_slen,
                 ptile_slen,
                 background = 686.): 
        
        self.tile_slen = tile_slen 
        self.ptile_slen = ptile_slen
        self.tiler = Tiler(tile_slen, ptile_slen).to(device)

        # the psf : this is an array of shape 
        # nbands x psf_slen x psf_slen
        self.n_bands = psf.shape[0]
        if self.n_bands > 1: 
            raise NotImplementedError()
        self.psf = self.tiler.fit_source_to_ptile(psf)
        self.psf_slen = self.psf.shape[-1]
        
        # sky background
        self.background = background
        
        # grid on which we render galaxies
        self.gal_slen = ptile_slen + (ptile_slen % 2 == 0)
        self.galaxy_mgrid = get_mgrid(self.gal_slen) 
        
        # this is so that the radius are in pixels
        self.galaxy_mgrid *= (self.gal_slen - 1) / 2 
        self.galaxy_mgrid = self.galaxy_mgrid.to(device)

    def render_stars(self, locs, fluxes, star_bool): 
        # locs: is (n_ptiles x max_num_stars x 2)
        # fluxes: Is (n_ptiles x max_stars x n_bands)
        # star_bool: Is (n_ptiles x max_stars x 1)
        
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]
        
        # all stars are just the PSF so we copy it.
        expanded_psf = self.psf.expand(n_ptiles,
                                       max_sources,
                                       self.n_bands,
                                       self.psf_slen,
                                       self.psf_slen)
        sources = expanded_psf * fluxes.unsqueeze(-1).unsqueeze(-1)
        sources *= star_bool.unsqueeze(-1).unsqueeze(-1)
        
        return self.tiler.render_tile(locs, sources)

    
    def render_galaxies(self, locs, galaxy_params, galaxy_bool):
        # galaxy params is a dictionary with 
        # keys giving the "fluxes"
        # theta : rotation angle 
        # ell : ellipticity 
        # r_dev : de Vaucouleurs galaxy radius 
        # r_exp : exponential galaxy radius 
        # all shapres are (n_ptiles x max_stars x 1)
        
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]
        
        flux = galaxy_params['flux'].view(n_ptiles * max_sources)
        theta = galaxy_params['theta'].view(n_ptiles * max_sources)
        ell = galaxy_params['ell'].view(n_ptiles * max_sources)
        r_dev = galaxy_params['r_dev'].view(n_ptiles * max_sources)
        r_exp = galaxy_params['r_exp'].view(n_ptiles * max_sources)
        p_dev = galaxy_params['p_dev'].view(n_ptiles * max_sources)
        
        # render centered galaxies
        centered_galaxies = \
            render_centered_galaxy(flux = flux, 
                                   theta = theta, 
                                   ell = ell, 
                                   r_dev = r_dev, 
                                   r_exp = r_exp,
                                   p_dev = p_dev, 
                                   galaxy_mgrid = self.galaxy_mgrid)
        
        # convolve w psf 
        centered_galaxies = _convolve_w_psf(centered_galaxies, self.psf)
                
        # reshape and turn off galaxies       
        centered_galaxies = centered_galaxies.view(n_ptiles, 
                                                   max_sources, 
                                                   self.n_bands,
                                                   self.gal_slen, 
                                                   self.gal_slen) * \
            galaxy_bool.unsqueeze(-1).unsqueeze(-1)       
        
        return self.tiler.render_tile(locs, centered_galaxies)
        