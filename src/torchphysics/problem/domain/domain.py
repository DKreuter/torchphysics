import abc
import numpy as np
import warnings


class Domain:
    def __init__(self, space, dim=None, tol=1e-06):
        self.space = space
        self.tol = tol
        if dim is None:
            self.dim = self.space.dim
        else:
            self.dim = dim

    @property
    def is_initialized(self):
        raise NotImplementedError

    def divide_points_to_space_variables(self, points):
        """Divides sample points of the form np.array(number_of_points, self.dim)
        to each variable of the given Space.

        Parameters
        ----------
        points: list, array
            The created sample/data points, need to fit the given dimension
        
        Returns
        -------
        dict
            A dictionary containing the input points but split up, to each 
            variable. E.g Space = R1('x')*R1('y') then the output would be
            output = {'x': points[:, 0:1], 'y': points[:, 1:2]}
        """
        output = {}
        current_dim = 0
        for vname in self.space:
            v_dim = self.space[vname]
            output[vname] = points[:, current_dim:current_dim+v_dim].astype(np.float32)
            current_dim += v_dim
        return output

    def return_space_variables_to_point_list(self, point_dic):
        """Concatenates sample points from a dict back to the form 
        np.array(number_of_points, self.dim)

        Parameters
        ----------
        point_dic: dic
            The dictionary of points 
            (most likely created with divide_points_to_space_variables)
        
        Returns
        -------
        points: array
            the point array of the form np.array(number_of_points, self.dim)
        """
        # if the points are not a dictonary just return
        # (not created with our sampling)
        if isinstance(point_dic, (list, np.ndarray)):
            return point_dic
        point_list = list(point_dic.values())
        return np.column_stack(point_list)

    @abc.abstractmethod
    def __add__(self, other):
        """Creates the union of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be united with the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __sub__(self, other):
        """Creates the cut of domain other from self.

        Parameters
        ----------
        other : Domain
            The other domain that should be cut off the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __and__(self, other):
        """Creates the intersection of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be intersected with the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_inside(self, points):
        """Checks for every point in points if it lays inside the domain.

        Parameters
        ----------
        points : list or array
            The list of diffrent or a single point that should be checked.
            E.g in 2D: points = [[2, 4], [9, 6], ....]

        Returns
        -------
        array
            A an array of the shape (len(points), 1) where every entry contains
            true if the point was inside or false if not.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def bounding_box(self):
        """Computes the bounds of the domain.

        Returns 
        list :
            A list with the length of 2*self.dim.
            It has the form [axis_1_min, axis_1_max, axis_2_min, axis_2_max, ...], 
            where min and max are the minimum and maximum value that the domain
            reaches in each dimension-axis.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_grid(self, n):
        """Greates a equdistant grid in the domain.

        Parameters
        ----------
        n : int
            The number of points that should be created.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_random_uniform(self, n):
        """Greates a random uniform points in the domain.

        Parameters
        ----------
        n : int
            The number of points that should be created.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def project_on_subspace(self, subspace):
        raise NotImplementedError

    def __mul__(self, other):
        return ProductDomain(self, other)

    @property
    def boundary(self):
        # Domain object of the boundary
        raise NotImplementedError

    @property
    def inner(self):
        # open domain
        raise NotImplementedError

    def _cut_points(self, n, points):
        """Deletes some random points, if more than n were sampled
        (can for example happen by grid-sampling).
        """
        if len(points) > n:
            index = np.random.choice(len(points), int(n), replace=False)
            return points[index]
        return points

    def _check_grid_enough_points(self, n, points):
        """Checks if there are not enough points for the grid.
        If not, add some random points.
        """
        if len(points) < n:
            new_points = self.sample_random_uniform(n-len(points))
            new_points = self.return_space_variables_to_point_list(new_points)
            points = np.append(points, new_points, axis=0)
        return points

    def _check_single_point(self, points):
        if len(np.shape(points)) <= 1:
            points = np.array([points])
        return points


class BoundaryDomain(Domain):
    def __init__(self, domain):
        super().__init__(domain.space, dim=domain.dim-1, tol=domain.tol)

    @abc.abstractmethod
    def normal(self, points):
        """Computes the normal vector at each point in points.
        
        Parameters
        ----------
        points : list or array
            A list of diffrent or a single point for which the normal vector 
            should be computed. The points must lay on the boundary of the domain.
            E.g in 2D: points = [[2, 4], [9, 6], ....]        

        Returns
        -------
        array
            The array is of the shape (len(points), self.dim) and contains the 
            normal vector at each entry from points.
        """
        pass


class ProductDomain(Domain):
    def __init__(self, domain_a, domain_b):
        self.domain_a = domain_a
        self.domain_b = domain_b
        if not self.domain_a.space.keys().isdisjoint(self.domain_b.space):
            warnings.warn("""Warning: The space of a ProductDomain will be the product
                of its factor domains spaces. This may lead to unexpected behaviour.""")
        space = self.domain_a.space * self.domain_b.space
        super().__init__(space=space,
                         dim=domain_a.dim + domain_b.dim,
                         tol=min(domain_a.tol, domain_b.tol))

    def project_on_subspace(self, subspace=None, **variables):
        assert subspace in self.space
        for domain in (self.domain_a, self.domain_b):
            if subspace in domain.space:
                return domain.project_on_subspace(subspace)
        return ProductDomain(self.domain_a.project_on_subspace(subspace & self.domain_a.space),
                             self.domain_b.project_on_subspace(subspace & self.domain_b.space))
        # TODO: fix this to keep order of dimensions with identical names


    @abc.abstractmethod
    def __add__(self, other):
        """Creates the union of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be united with the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __sub__(self, other):
        """Creates the cut of other from self.

        Parameters
        ----------
        other : Domain
            The other domain that should be cut off the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __and__(self, other):
        """Creates the intersection of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be intersected with the domain.
            Has to be of the same dimension.
        """
        return ProductDomain(self.domain_a & other, self.domain_b & other)

    @abc.abstractmethod
    def is_inside(self, points):
        return 

    @abc.abstractmethod
    def bounding_box(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_grid(self, n):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_random_uniform(self, n):
        raise NotImplementedError

    @property
    def boundary(self):
        # Domain object of the boundary
        return ProductDomain(self.domain_a.boundary, self.domain_b.boundary)

    @property
    def inner(self):
        # open domain
        return ProductDomain(self.domain_a.inner, self.domain_b.inner)
