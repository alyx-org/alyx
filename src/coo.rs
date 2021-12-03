use std::mem::size_of;

/// Coordinate (COO) format sparse matrix.
///
/// # Format
///
/// The coordinate storage format stores tuples `(row, col, value)` for each
/// non-zero element of the matrix.
///
/// ## Properties
///
/// - The coordinate format is intented for incremental matrix constructions.
/// - The coordinate format **allows** duplicates.
///
/// ## Storage
///
/// Let `A` a 3-by-4 (real) matrix with 6 non-zero entries:
///
/// ```text
///     | 0 1 0 2 |
/// A = | 0 0 3 0 |
///     | 4 5 6 0 |
/// ```
///
/// In arbitrary order, 'A' may be stored as follows:
///
/// ```text
/// (row, col, val) idx
/// (0, 1, 1.0)     [0]
/// (1, 2, 3.0)     [1]
/// (2, 1, 5.0)     [2]
/// (0, 3, 2.0)     [3]
/// (2, 0, 4.0)     [4]
/// (2, 2, 6.0)     [5]
/// ```
///
/// In row major order, `A` is stored as follows:
///
/// ```text
/// (row, col, val) idx
/// (0, 1, 1.0)     [0]
/// (0, 3, 2.0)     [1]
/// (1, 2, 3.0)     [2]
/// (2, 0, 4.0)     [3]
/// (2, 1, 5.0)     [4]
/// (2, 2, 6.0)     [5]
///  ^
///  | major
/// ```
///
/// In column major order, `A` is stored as follows:
///
/// ```text
/// (row, col, val) idx
/// (2, 0, 4.0)     [0]
/// (0, 1, 1.0)     [1]
/// (2, 1, 5.0)     [2]
/// (1, 2, 3.0)     [3]
/// (2, 2, 6.0)     [4]
/// (0, 3, 2.0)     [5]
///     ^
///     | major
/// ```
///
/// ## Constructors
///
/// `CooMat` provides multiple constructors:
///
/// ```
/// # use alyx::CooMat;
///
/// // Create an empty COO matrix
/// let rows = 2;
/// let cols = 3;
/// let matrix = CooMat::<f64>::new(rows, cols);
///
/// // Create an empty COO matrix with initial capacity
/// let rows = 2;
/// let cols = 3;
/// let capacity = 6;
/// let matrix = CooMat::<f64>::with_capacity(rows, cols, capacity);
///
/// // Create a COO matrix with initial entries from entry vector
/// let rows = 2;
/// let cols = 2;
/// let entries = vec![(0, 0, 1.0), (1, 1, 2.0)];
/// let matrix = CooMat::with_entries(rows, cols, entries);
///
/// // Create a COO matrix with initial entries from triplet vectors
/// let rows = 2;
/// let cols = 2;
/// let rowind = vec![0, 1];
/// let colind = vec![0, 1];
/// let values = vec![1.0, 2.0];
/// let matrix = CooMat::with_triplets(rows, cols, rowind, colind, values);
/// ```
///
/// ## Entry get/insertion/removal/clear
///
/// `CooMat` is intented for incremental matrix constructions and provides corresponding methods:
///
/// ```
/// use alyx::CooMat;
///
/// let mut matrix = CooMat::with_capacity(3, 3, 9);
/// // matrix:
/// // | 0 0 0 |
/// // | 0 0 0 |
/// // | 0 0 0 |
///
/// // Insert new entries one by one
/// matrix.push(0, 0, 1.0);
/// matrix.push(0, 1, 2.0);
/// matrix.push(1, 0, 3.0);
/// matrix.push(1, 1, 4.0);
/// // matrix:
/// // | 1 2 0 |
/// // | 3 4 0 |
/// // | 0 0 0 |
/// // entries:
/// // (0, 0, 1.0) [0]
/// // (0, 1, 2.0) [1]
/// // (1, 0, 3.0) [2]
/// // (1, 1, 4.0) [3]
/// assert_eq!(matrix.len(), 4);
///
/// // Get an immutable reference to an entry
/// let entry = matrix.get(0);
/// assert_eq!(entry, Some((&0, &0, &1.0)));
///
/// // Get a mutable reference to an entry and modify it
/// // (only the value is mutable)
/// if let Some((r, c, v)) = matrix.get_mut(0) {
///     *v *= -1.0;
/// }
/// // matrix:
/// // |-1 2 0 |
/// // | 3 4 0 |
/// // | 0 0 0 |
/// // entries:
/// // (0, 0,-1.0) [0]
/// // (0, 1, 2.0) [1]
/// // (1, 0, 3.0) [2]
/// // (1, 1, 4.0) [3]
/// let entry = matrix.get(0);
/// assert_eq!(entry, Some((&0, &0, &-1.0)));
///
/// // Extend the matrix with new entries
/// let entries = vec![
///     (0, 2, 5.0),
///     (1, 2, 6.0),
///     (2, 1, 8.0),
///     (2, 2, 9.0),
/// ];
/// matrix.extend(entries);
/// // matrix:
/// // |-1 2 5 |
/// // | 3 4 6 |
/// // | 0 8 9 |
/// // entries:
/// // (0, 0,-1.0) [0]
/// // (0, 1, 2.0) [1]
/// // (1, 0, 3.0) [2]
/// // (1, 1, 4.0) [3]
/// // (0, 2, 5.0) [4] <|
/// // (1, 2, 6.0) [5] <|
/// // (2, 1, 8.0) [6] <|
/// // (2, 2, 9.0) [7] <|- entries added
/// assert_eq!(matrix.len(), 8);
///
/// // Insert new entry at specified index
/// matrix.insert(6, 2, 0, 7.0);
/// // matrix:
/// // |-1 2 5 |
/// // | 3 4 6 |
/// // | 7 8 9 |
/// // entries:
/// // (0, 0,-1.0) [0]
/// // (0, 1, 2.0) [1]
/// // (1, 0, 3.0) [2]
/// // (1, 1, 4.0) [3]
/// // (0, 2, 5.0) [4]
/// // (1, 2, 6.0) [5]
/// // (2, 0, 7.0) [6] <- entry inserted
/// // (2, 1, 8.0) [7] <|
/// // (2, 2, 9.0) [8] <|- indices shifted
/// assert_eq!(matrix.len(), 9);
/// assert_eq!(matrix.get(6), Some((&2, &0, &7.0)));
///
/// // Remove last entry
/// assert_eq!(matrix.pop(), Some((2, 2, 9.0)));
/// // matrix:
/// // |-1 2 5 |
/// // | 3 4 6 |
/// // | 7 8 0 |
/// // entries:
/// // (0, 0,-1.0) [0]
/// // (0, 1, 2.0) [1]
/// // (1, 0, 3.0) [2]
/// // (1, 1, 4.0) [3]
/// // (0, 2, 5.0) [4]
/// // (1, 2, 6.0) [5]
/// // (2, 0, 7.0) [6]
/// // (2, 1, 8.0) [7]
/// // (2, 2, 9.0) [x] <- entry removed
/// assert_eq!(matrix.len(), 8);
///
/// // Remove a specific entry
/// let entry = matrix.remove(1);
/// assert_eq!(entry, (0, 1, 2.0));
/// // matrix:
/// // |-1 0 5 |
/// // | 3 4 6 |
/// // | 7 8 0 |
/// // entries:
/// // (0, 0,-1.0) [0]
/// // (0, 1, 2.0) [x] <- entry removed
/// // (1, 0, 3.0) [1] <|
/// // (1, 1, 4.0) [2] <|
/// // (0, 2, 5.0) [3] <|
/// // (1, 2, 6.0) [4] <|
/// // (2, 0, 7.0) [5] <|
/// // (2, 1, 8.0) [6] <|- indices shifted
/// assert_eq!(matrix.len(), 7);
///
/// // Clear matrix
/// matrix.clear();
/// assert!(matrix.is_empty())
/// ```
///
/// ## Capacity and Length
///
/// The matrix *capacity* corresponds to the amount of space allocated for the
/// matrix entries unlike the matrix *length* which corresponds to the number
/// of entries in the matrix.
/// Initially for empty matrices, no memory is allocated.
/// If entries are inserted, matrix capacity will grow accordingly.
/// The growth strategy is *unspecified behavior* and no guarentees are made.
///
/// `CooMat` provides multiple methods to manage matrix capacity and length:
///
/// ```
/// # use alyx::CooMat;
///
/// let mut matrix: CooMat<f64> = CooMat::new(2, 2);
/// // Initially capacity and length are 0
/// assert_eq!(matrix.capacity(), 0);
/// assert_eq!(matrix.len(), 0);
///
/// // Inserting an entry allocate space for at least one entry
/// matrix.push(0, 0, 1.0);
/// assert!(matrix.capacity() >= 1);
/// assert_eq!(matrix.len(), 1);
///
/// // Inserting additional entries may reallocate
/// matrix.push(1, 1, 2.0);
/// assert!(matrix.capacity() >= 2);
/// assert_eq!(matrix.len(), 2);
///
/// // To prevent reallocation, capacity can be adjusted at construction time
/// let mut matrix: CooMat<f64> = CooMat::with_capacity(2, 2, 4);
/// assert_eq!(matrix.capacity(), 4);
/// assert!(matrix.is_empty());
///
/// // Pushing values will not reallocate
/// matrix.push(0, 0, 1.0);
/// assert_eq!(matrix.capacity(), 4);
/// assert_eq!(matrix.len(), 1);
///
/// // Additional capacity can be requested after construction
/// let mut matrix: CooMat<f64> = CooMat::with_capacity(2, 2, 1);
/// assert_eq!(matrix.capacity(), 1);
/// matrix.reserve(4);
/// assert!(matrix.capacity() >= 4);
///
/// // Capacity can be shrunk to fit actual number of entries
/// let mut matrix: CooMat<f64> = CooMat::with_capacity(3, 3, 9);
/// assert_eq!(matrix.capacity(), 9);
/// matrix.push(1, 1, 1.0);
/// matrix.shrink();
/// assert!(matrix.capacity() < 9);
/// ```
///
/// ## Iterators
///
/// `CooMat` also provides convenient iterators (`Iter`, `IterMut` and
/// `IntoIter`):
///
/// ```
/// # use alyx::CooMat;
/// let entries = vec![
///     (0, 0, 1.0),
///     (0, 1, 2.0),
///     (1, 0, 3.0),
///     (1, 1, 4.0),
/// ];
/// let mut matrix = CooMat::with_entries(2, 2, entries);
///
/// // Immutable iterator over entries
/// let mut iter = matrix.iter();
/// assert_eq!(iter.next(), Some((&0, &0, &1.0)));
/// assert_eq!(iter.next(), Some((&0, &1, &2.0)));
/// assert_eq!(iter.next(), Some((&1, &0, &3.0)));
/// assert_eq!(iter.next(), Some((&1, &1, &4.0)));
/// assert_eq!(iter.next(), None);
///
/// // Mutable iterator over entries
/// let mut iter = matrix.iter_mut();
/// for (r, c, v) in iter {
///     *v *= 2.0;
/// }
/// let mut iter = matrix.iter();
/// assert_eq!(iter.next(), Some((&0, &0, &2.0)));
/// assert_eq!(iter.next(), Some((&0, &1, &4.0)));
/// assert_eq!(iter.next(), Some((&1, &0, &6.0)));
/// assert_eq!(iter.next(), Some((&1, &1, &8.0)));
/// assert_eq!(iter.next(), None);
///
/// // Turn matrix into iterator
/// for (r, c, v) in matrix {
///     println!("row = {}, col = {}, value = {}", r, c, v);
/// }
/// ```
#[derive(Debug)]
pub struct CooMat<T> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) entries: Vec<(usize, usize, T)>,
}

impl<T> CooMat<T> {
    /// Creates a coordinate format sparse matrix with specified shape
    /// `(rows, cols)`.
    ///
    /// The created matrix has following properties:
    /// - the matrix is empty (`matrix.len() == 0`)
    /// - the matrix does **not** allocate memory (`matrix.capacity() == 0`)
    /// - the matrix will **not** allocate memory before any insert/push/extend
    ///   operation
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// // With type annotation ...
    /// let matrix: CooMat<f64> = CooMat::new(2, 2);
    /// assert_eq!(matrix.shape(), (2, 2));
    /// assert!(matrix.is_empty());
    /// assert_eq!(matrix.capacity(), 0);
    ///
    /// // ... or with turbofish operator
    /// let matrix = CooMat::<f64>::new(3, 4);
    /// assert_eq!(matrix.shape(), (3, 4));
    /// assert!(matrix.is_empty());
    /// assert_eq!(matrix.capacity(), 0);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `size_of::<T> == 0`
    /// - `rows == 0`
    /// - `cols == 0`
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(size_of::<T>() != 0);
        assert!(rows > 0);
        assert!(cols > 0);
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    /// Creates a coordinate format sparse matrix with specified shape
    /// `(rows, cols)` and capacity.
    ///
    /// The created matrix has following properties:
    /// - the matrix is empty (`matrix.len() == 0`)
    /// - the matrix allocates memory for at least `capacity` entries
    ///   (`matrix.capacity() >= capacity`)
    /// - the matrix will not allocate if `capacity == 0`
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let matrix = CooMat::<f64>::with_capacity(2, 2, 4);
    /// assert_eq!(matrix.shape(), (2, 2));
    /// assert!(matrix.is_empty());
    /// assert_eq!(matrix.capacity(), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `size_of::<T> == 0`
    /// - `rows == 0`
    /// - `cols == 0`
    /// - the allocation size exceeds `isize::MAX` bytes
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        assert!(size_of::<T>() != 0);
        assert!(rows > 0);
        assert!(cols > 0);
        Self {
            rows,
            cols,
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Creates a coordinate format sparse matrix with specified shape
    /// `(rows, cols)` and entries.
    ///
    /// The created matrix has following properties:
    /// - the matrix is filled with `entries`
    /// - the matrix allocates memory for at least `entries.len()` entries
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let matrix = CooMat::with_entries(2, 2, entries);
    /// assert_eq!(matrix.shape(), (2, 2));
    /// assert_eq!(matrix.len(), 4);
    /// assert!(matrix.capacity() >= 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `size_of::<T> == 0`
    /// - `rows == 0`
    /// - `cols == 0`
    /// - for any entry `(row, col, val)`: `row >= rows` or `col >= cols`
    pub fn with_entries(
        rows: usize,
        cols: usize,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Self {
        assert!(size_of::<T>() != 0);
        assert!(rows > 0);
        assert!(cols > 0);
        let entries: Vec<_> = entries.into_iter().collect();
        for (row, col, _) in entries.iter() {
            assert!(*row < rows);
            assert!(*col < cols);
        }
        Self {
            rows,
            cols,
            entries,
        }
    }

    /// Creates a coordinate format sparse matrix with specified shape
    /// `(rows, cols)`, capacity and entries.
    ///
    /// The created matrix has following properties:
    /// - the matrix is filled with `entries`
    /// - the matrix allocates memory for at least
    ///   `max(capacity, entries.len())` entries
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let matrix = CooMat::with_capacity_and_entries(2, 2, 4, entries);
    /// assert_eq!(matrix.shape(), (2, 2));
    /// assert_eq!(matrix.len(), 4);
    /// assert_eq!(matrix.capacity(), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `size_of::<T> == 0`
    /// - `rows == 0`
    /// - `cols == 0`
    /// - for any entry `(row, col, val)`: `row >= rows` or `col >= cols`
    /// - the allocation size exceeds `isize::MAX` bytes
    pub fn with_capacity_and_entries(
        rows: usize,
        cols: usize,
        capacity: usize,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Self {
        assert!(size_of::<T>() != 0);
        assert!(rows > 0);
        assert!(cols > 0);
        let iter = entries.into_iter();
        let mut entries = Vec::with_capacity(capacity);
        for entry @ (row, col, _) in iter {
            assert!(row < rows);
            assert!(col < cols);
            entries.push(entry);
        }
        Self {
            rows,
            cols,
            entries,
        }
    }

    /// Creates a coordinate format sparse matrix with specified shape
    /// `(rows, cols)` and triplets.
    ///
    /// The created matrix has following properties:
    /// - the matrix is filled with `values.len()` entries
    /// - the matrix allocates memory for at least `values.len()` entries
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let rowind = vec![0, 0, 1, 1];          // entry row indices
    /// let colind = vec![0, 1, 0, 1];          // entry column indices
    /// let values = vec![1.0, 2.0, 3.0, 4.0];  // entry values
    /// let matrix = CooMat::with_triplets(2, 2, rowind, colind, values);
    /// assert_eq!(matrix.shape(), (2, 2));
    /// assert_eq!(matrix.len(), 4);
    /// assert!(matrix.capacity() >= 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `size_of::<T> == 0`
    /// - `rows == 0`
    /// - `cols == 0`
    /// - for any entry `(row, col, val)`: `row >= rows` or `col >= cols`
    /// - `rowind`, `colind` and `values` length differ
    pub fn with_triplets(
        rows: usize,
        cols: usize,
        rowind: impl IntoIterator<Item = usize>,
        colind: impl IntoIterator<Item = usize>,
        values: impl IntoIterator<Item = T>,
    ) -> Self {
        assert!(size_of::<T>() != 0);
        assert!(rows > 0);
        assert!(cols > 0);
        let mut rowind = rowind.into_iter();
        let mut colind = colind.into_iter();
        let values = values.into_iter();
        let capacity = match values.size_hint() {
            (_, Some(high)) => high,
            (low, None) => low,
        };
        let mut entries = Vec::with_capacity(capacity);
        for value in values {
            let row = rowind.next().unwrap();
            let col = colind.next().unwrap();
            assert!(row < rows);
            assert!(col < cols);
            entries.push((row, col, value));
        }
        assert!(rowind.next().is_none());
        assert!(colind.next().is_none());
        entries.shrink_to_fit();
        CooMat {
            rows,
            cols,
            entries,
        }
    }

    /// Creates a coordinate format sparse matrix with specified shape
    /// `(rows, cols)`, capacity and triplets.
    ///
    /// The created matrix has following properties:
    /// - the matrix is filled with `values.len()` entries
    /// - the matrix allocates memory for at least
    ///   `max(capacity, values.len())` entries
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let rowind = vec![0, 0, 1, 1];          // entry row indices
    /// let colind = vec![0, 1, 0, 1];          // entry column indices
    /// let values = vec![1.0, 2.0, 3.0, 4.0];  // entry values
    /// let matrix = CooMat::with_capacity_and_triplets(
    ///     2, 2, 4,
    ///     rowind,
    ///     colind,
    ///     values
    /// );
    /// assert_eq!(matrix.shape(), (2, 2));
    /// assert_eq!(matrix.len(), 4);
    /// assert_eq!(matrix.capacity(), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `size_of::<T> == 0`
    /// - `rows == 0`
    /// - `cols == 0`
    /// - for any entry `(row, col, val)`: `row >= rows` or `col >= cols`
    /// - `rowind`, `colind` and `values` length differ
    /// - the allocation size exceeds `isize::MAX` bytes
    pub fn with_capacity_and_triplets(
        rows: usize,
        cols: usize,
        capacity: usize,
        rowind: impl IntoIterator<Item = usize>,
        colind: impl IntoIterator<Item = usize>,
        values: impl IntoIterator<Item = T>,
    ) -> Self {
        assert!(size_of::<T>() != 0);
        assert!(rows > 0);
        assert!(cols > 0);
        let mut rowind = rowind.into_iter();
        let mut colind = colind.into_iter();
        let values = values.into_iter();
        let mut entries = Vec::with_capacity(capacity);
        for value in values {
            let row = rowind.next().expect("invalid matrix");
            let col = colind.next().expect("invalid matrix");
            assert!(row < rows);
            assert!(col < cols);
            entries.push((row, col, value));
        }
        assert!(rowind.next().is_none());
        assert!(colind.next().is_none());
        CooMat {
            rows,
            cols,
            entries,
        }
    }

    /// Returns the number of rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let matrix = CooMat::<f64>::new(1, 2);
    /// assert_eq!(matrix.rows(), 1);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let matrix = CooMat::<f64>::new(1, 2);
    /// assert_eq!(matrix.cols(), 2);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the shape `(rows, cols)` of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let matrix = CooMat::<f64>::new(1, 2);
    /// assert_eq!(matrix.shape(), (1, 2));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of entries of the matrix.
    ///
    /// This number is **not** the number of non-zero elements in the matrix
    /// because duplicates are allowed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let matrix = CooMat::with_entries(2, 2, entries);
    /// assert_eq!(matrix.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the matrix contains no entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let matrix = CooMat::with_entries(2, 2, entries);
    /// let empty = CooMat::<f64>::new(2, 2);
    /// assert!(!matrix.is_empty());
    /// assert!(empty.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the capacity of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let matrix = CooMat::<f64>::with_capacity(1, 1, 42);
    /// assert_eq!(matrix.capacity(), 42);
    /// ```
    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Reserve capacity for additional entries.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let mut matrix = CooMat::<f64>::new(1, 1);
    /// assert_eq!(matrix.capacity(), 0);
    /// matrix.reserve(10);
    /// assert!(matrix.capacity() >= 10);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size exceeds `isize::MAX` bytes
    pub fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional)
    }

    /// Shrink matrix capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let mut matrix = CooMat::<f64>::with_capacity(1, 1, 42);
    /// assert_eq!(matrix.capacity(), 42);
    /// matrix.shrink();
    /// assert!(matrix.capacity() <= 42);
    /// ```
    pub fn shrink(&mut self) {
        self.entries.shrink_to_fit()
    }

    /// Shortens the matrix to `len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let mut matrix = CooMat::with_entries(2, 2, entries);
    /// assert_eq!(matrix.len(), 4);
    /// matrix.truncate(4);
    /// assert_eq!(matrix.len(), 4);
    /// matrix.truncate(2);
    /// assert_eq!(matrix.len(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `len > self.len()`
    pub fn truncate(&mut self, len: usize) {
        assert!(len <= self.len());
        self.entries.truncate(len)
    }

    /// Clears the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let mut matrix = CooMat::with_entries(2, 2, entries);
    /// matrix.clear();
    /// assert!(matrix.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.entries.clear()
    }

    /// Returns an immutable reference to the entry at specified index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![(0, 0, 1.0)];
    /// let matrix = CooMat::with_entries(1, 1, entries);
    /// assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
    /// ```
    pub fn get(&self, index: usize) -> Option<(&usize, &usize, &T)> {
        self.entries.get(index).map(|(r, c, v)| (r, c, v))
    }

    /// Returns a mutable reference to the entry at specified index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![(0, 0, 1.0)];
    /// let mut matrix = CooMat::with_entries(1, 1, entries);
    /// if let Some((_, _, v)) = matrix.get_mut(0) {
    ///     *v *= 2.0;
    /// }
    /// assert_eq!(matrix.get_mut(0), Some((&0, &0, &mut 2.0)));
    /// ```
    pub fn get_mut(&mut self, index: usize) -> Option<(&usize, &usize, &mut T)> {
        self.entries.get_mut(index).map(|(r, c, v)| (&*r, &*c, v))
    }

    /// Push an entry into this matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let mut matrix = CooMat::new(1, 1);
    /// matrix.push(0, 0, 1.0);
    /// assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `row >= self.rows`
    /// - `col >= self.cols`
    /// - the new allocation size exceeds `isize::MAX` bytes
    pub fn push(&mut self, row: usize, col: usize, val: T) {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.entries.push((row, col, val))
    }

    /// Pop entry from the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![(0, 0, 1.0)];
    /// let mut matrix = CooMat::with_entries(1, 1, entries);
    /// assert_eq!(matrix.pop(), Some((0, 0, 1.0)));
    /// assert_eq!(matrix.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<(usize, usize, T)> {
        self.entries.pop()
    }

    /// Insert an entry into the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let mut matrix = CooMat::new(1, 1);
    /// matrix.insert(0, 0, 0, 1.0);
    /// assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `row >= self.rows`
    /// - `col >= self.cols`
    /// - `index > self.len()`
    pub fn insert(&mut self, index: usize, row: usize, col: usize, val: T) {
        assert!(row < self.rows);
        assert!(col < self.cols);
        assert!(index <= self.len());
        self.entries.insert(index, (row, col, val))
    }

    /// Extend matrix entries.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let mut matrix: CooMat<f64> = CooMat::new(2, 2);
    /// assert_eq!(matrix.len(), 0);
    /// matrix.extend(entries);
    /// assert_eq!(matrix.len(), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `row >= self.rows`
    /// - `col >= self.cols`
    pub fn extend<I: IntoIterator<Item = (usize, usize, T)>>(&mut self, iter: I) {
        for (row, col, val) in iter {
            assert!(row < self.rows);
            assert!(col < self.cols);
            self.entries.push((row, col, val));
        }
    }

    /// Remove an entry from the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![(0, 0, 1.0)];
    /// let mut matrix = CooMat::with_entries(1, 1, entries);
    /// assert_eq!(matrix.remove(0), (0, 0, 1.0));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// `index >= self.len()`
    pub fn remove(&mut self, index: usize) -> (usize, usize, T) {
        self.entries.remove(index)
    }

    /// Returns an iterator visiting all entries of the matrix.
    /// The iterator element type is `(&'a usize, &'a usize, &'a T)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let matrix = CooMat::with_entries(2, 2, entries);
    ///
    /// let mut iter = matrix.iter();
    /// assert_eq!(iter.next(), Some((&0, &0, &1.0)));
    /// assert_eq!(iter.next(), Some((&0, &1, &2.0)));
    /// assert_eq!(iter.next(), Some((&1, &0, &3.0)));
    /// assert_eq!(iter.next(), Some((&1, &1, &4.0)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter {
            inner: self.entries.iter(),
        }
    }

    /// Returns an iterator visiting all mutable entries of the matrix.
    /// The iterator element type is `(&'a usize, &'a usize, &'a mut T)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use alyx::CooMat;
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (0, 1, 2.0),
    ///     (1, 0, 3.0),
    ///     (1, 1, 4.0),
    /// ];
    /// let mut matrix = CooMat::with_entries(2, 2, entries);
    ///
    /// let mut iter = matrix.iter_mut();
    /// for (_, _, v) in iter {
    ///     *v *= 2.0;
    /// }
    ///
    /// let mut iter = matrix.iter_mut();
    /// assert_eq!(iter.next(), Some((&0, &0, &mut 2.0)));
    /// assert_eq!(iter.next(), Some((&0, &1, &mut 4.0)));
    /// assert_eq!(iter.next(), Some((&1, &0, &mut 6.0)));
    /// assert_eq!(iter.next(), Some((&1, &1, &mut 8.0)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            inner: self.entries.iter_mut(),
        }
    }
}

impl<T> IntoIterator for CooMat<T> {
    type Item = (usize, usize, T);

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.entries.into_iter(),
        }
    }
}

/// Immutable coordinate sparse matrix iterator.
///
/// This iterator is created by the [iter()](CooMat::iter()) method on
/// coordinate sparse matrix.
#[derive(Clone, Debug)]
pub struct Iter<'iter, T: 'iter> {
    inner: std::slice::Iter<'iter, (usize, usize, T)>,
}

impl<'iter, T: 'iter> Iterator for Iter<'iter, T> {
    type Item = (&'iter usize, &'iter usize, &'iter T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(r, c, v)| (r, c, v))
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(r, c, v)| (r, c, v))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

/// Mutable coordinate sparse matrix iterator.
///
/// This iterator is created by the [iter_mut()](CooMat::iter_mut()) method on
/// coordinate sparse matrix.
#[derive(Debug)]
pub struct IterMut<'iter, T: 'iter> {
    inner: std::slice::IterMut<'iter, (usize, usize, T)>,
}

impl<'iter, T: 'iter> Iterator for IterMut<'iter, T> {
    type Item = (&'iter usize, &'iter usize, &'iter mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(r, c, v)| (&*r, &*c, v))
    }
}

impl<T> DoubleEndedIterator for IterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(r, c, v)| (&*r, &*c, v))
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

/// An iterator that moves out of a coordinate sparse matrix.
///
/// # Examples
///
/// ```
/// # use alyx::CooMat;
/// let start = vec![
///     (0, 0, 1.0),
///     (0, 1, 2.0),
///     (1, 0, 3.0),
///     (1, 1, 4.0),
/// ];
/// let matrix = CooMat::with_entries(2, 2, start.clone());
/// let end: Vec<_> = matrix.into_iter().collect();
/// assert_eq!(start, end);
/// ```
#[derive(Debug)]
pub struct IntoIter<T> {
    inner: std::vec::IntoIter<(usize, usize, T)>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = (usize, usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    #[should_panic]
    fn new_zst() {
        CooMat::<()>::new(1, 1);
    }

    #[test]
    fn new_annotation() {
        let _: CooMat<f64> = CooMat::new(1, 2);
    }

    #[test]
    fn new_fish() {
        CooMat::<f64>::new(2, 1);
    }

    #[test]
    fn new_shape() {
        let matrix = CooMat::<f64>::new(1, 2);
        assert_eq!(matrix.shape(), (1, 2));
    }

    #[test]
    fn new_length() {
        let matrix = CooMat::<f64>::new(1, 1);
        assert_eq!(matrix.entries.len(), 0)
    }

    #[test]
    fn new_capacity() {
        let matrix = CooMat::<f64>::new(1, 1);
        assert_eq!(matrix.entries.capacity(), 0);
    }

    #[test]
    #[should_panic]
    fn with_capacity_zst() {
        CooMat::<()>::new(1, 1);
    }

    #[test]
    fn with_capacity_annotation() {
        let _: CooMat<f64> = CooMat::with_capacity(1, 2, 3);
    }

    #[test]
    fn with_capacity_fish() {
        CooMat::<f64>::with_capacity(2, 1, 3);
    }

    #[test]
    fn with_capacity_shape() {
        let matrix = CooMat::<f64>::with_capacity(1, 2, 3);
        assert_eq!(matrix.shape(), (1, 2));
    }

    #[test]
    fn with_capacity_length() {
        let matrix = CooMat::<f64>::with_capacity(1, 1, 1);
        assert_eq!(matrix.entries.len(), 0)
    }

    #[test]
    fn with_capacity_capacity() {
        let matrix = CooMat::<f64>::with_capacity(1, 1, 1);
        assert_eq!(matrix.entries.capacity(), 1);
    }

    #[test]
    #[should_panic]
    fn with_entries_zst() {
        CooMat::<()>::with_entries(1, 1, vec![]);
    }

    #[test]
    fn with_entries() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = CooMat::with_entries(1, 1, entries);
        assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
        assert_eq!(matrix.get(1), None);
    }

    #[test]
    fn with_entries_order() {
        let entries = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let matrix = CooMat::with_entries(2, 2, entries);
        let mut iter = matrix.iter();
        assert_eq!(iter.next(), Some((&0, &0, &1.0)));
        assert_eq!(iter.next(), Some((&0, &1, &2.0)));
        assert_eq!(iter.next(), Some((&1, &0, &3.0)));
        assert_eq!(iter.next(), Some((&1, &1, &4.0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn with_entries_length() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = CooMat::with_entries(1, 1, entries);
        assert_eq!(matrix.entries.len(), 1)
    }

    #[test]
    fn with_entries_capacity() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = CooMat::with_entries(1, 1, entries);
        assert_eq!(matrix.entries.capacity(), 1)
    }

    #[test]
    #[should_panic]
    fn with_entries_invalid_row() {
        let entries = vec![(1, 0, 1.0)];
        CooMat::with_entries(1, 1, entries);
    }

    #[test]
    #[should_panic]
    fn with_entries_invalid_col() {
        let entries = vec![(0, 1, 1.0)];
        CooMat::with_entries(1, 1, entries);
    }

    #[test]
    #[should_panic]
    fn with_capacity_and_entries_zst() {
        let entries = vec![(0, 0, ())];
        CooMat::with_capacity_and_entries(1, 1, 1, entries);
    }

    #[test]
    fn with_capacity_and_entries() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = CooMat::with_capacity_and_entries(1, 1, 1, entries);
        assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
        assert_eq!(matrix.get(1), None);
    }

    #[test]
    fn with_capacity_and_entries_order() {
        let entries = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let matrix = CooMat::with_capacity_and_entries(2, 2, 4, entries);
        let mut iter = matrix.iter();
        assert_eq!(iter.next(), Some((&0, &0, &1.0)));
        assert_eq!(iter.next(), Some((&0, &1, &2.0)));
        assert_eq!(iter.next(), Some((&1, &0, &3.0)));
        assert_eq!(iter.next(), Some((&1, &1, &4.0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn with_capacity_and_entries_length() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = CooMat::with_capacity_and_entries(1, 1, 1, entries);
        assert_eq!(matrix.entries.len(), 1)
    }

    #[test]
    fn with_capacity_and_entries_capacity() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = CooMat::with_capacity_and_entries(1, 1, 1, entries);
        assert_eq!(matrix.entries.capacity(), 1)
    }

    #[test]
    #[should_panic]
    fn with_capacity_and_entries_invalid_row() {
        let entries = vec![(1, 0, 1.0)];
        CooMat::with_capacity_and_entries(1, 1, 1, entries);
    }

    #[test]
    #[should_panic]
    fn with_capacity_and_entries_invalid_col() {
        let entries = vec![(0, 1, 1.0)];
        CooMat::with_capacity_and_entries(1, 1, 1, entries);
    }

    #[test]
    #[should_panic]
    fn with_triplets_zst() {
        CooMat::<()>::with_triplets(1, 1, vec![0], vec![0], vec![()]);
    }

    #[test]
    fn with_triplets() {
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![1.0];
        let matrix = CooMat::with_triplets(1, 1, rowind, colind, values);
        assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
        assert_eq!(matrix.get(1), None);
    }

    #[test]
    fn with_triplets_order() {
        let rowind = vec![0, 0, 1, 1];
        let colind = vec![0, 1, 0, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = CooMat::with_triplets(2, 2, rowind, colind, values);
        let mut iter = matrix.iter();
        assert_eq!(iter.next(), Some((&0, &0, &1.0)));
        assert_eq!(iter.next(), Some((&0, &1, &2.0)));
        assert_eq!(iter.next(), Some((&1, &0, &3.0)));
        assert_eq!(iter.next(), Some((&1, &1, &4.0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn with_triplets_length() {
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![1.0];
        let matrix = CooMat::with_triplets(1, 1, rowind, colind, values);
        assert_eq!(matrix.entries.len(), 1);
    }

    #[test]
    fn with_triplets_capacity() {
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![1.0];
        let matrix = CooMat::with_triplets(1, 1, rowind, colind, values);
        assert_eq!(matrix.entries.capacity(), 1);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_row() {
        let rowind = vec![1];
        let colind = vec![0];
        let values = vec![1.0];
        CooMat::with_triplets(1, 1, rowind, colind, values);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_col() {
        let rowind = vec![0];
        let colind = vec![1];
        let values = vec![1.0];
        CooMat::with_triplets(1, 1, rowind, colind, values);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_length() {
        let rowind = vec![0; 2];
        let colind = vec![1; 1];
        let values = vec![1.0; 3];
        CooMat::with_triplets(1, 1, rowind, colind, values);
    }

    #[test]
    #[should_panic]
    fn with_capacity_and_triplets_zst() {
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![()];
        CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
    }

    #[test]
    fn with_capacity_and_triplets() {
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![1.0];
        let matrix = CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
        assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
        assert_eq!(matrix.get(1), None);
    }

    #[test]
    fn with_capacity_and_triplets_order() {
        let rowind = vec![0, 0, 1, 1];
        let colind = vec![0, 1, 0, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = CooMat::with_capacity_and_triplets(2, 2, 4, rowind, colind, values);
        let mut iter = matrix.iter();
        assert_eq!(iter.next(), Some((&0, &0, &1.0)));
        assert_eq!(iter.next(), Some((&0, &1, &2.0)));
        assert_eq!(iter.next(), Some((&1, &0, &3.0)));
        assert_eq!(iter.next(), Some((&1, &1, &4.0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn with_capacity_and_triplets_length() {
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![1.0];
        let matrix = CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
        assert_eq!(matrix.entries.len(), 1);
    }

    #[test]
    fn with_capacity_and_triplets_capacity() {
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![1.0];
        let matrix = CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
        assert_eq!(matrix.entries.capacity(), 1);
    }

    #[test]
    #[should_panic]
    fn with_capacity_and_triplets_invalid_row() {
        let rowind = vec![1];
        let colind = vec![0];
        let values = vec![1.0];
        CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
    }

    #[test]
    #[should_panic]
    fn with_capacity_and_triplets_invalid_col() {
        let rowind = vec![0];
        let colind = vec![1];
        let values = vec![1.0];
        CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
    }

    #[test]
    #[should_panic]
    fn with_capacity_and_triplets_invalid_length() {
        let rowind = vec![0; 2];
        let colind = vec![1; 1];
        let values = vec![1.0; 3];
        CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
    }

    #[test]
    fn rows() {
        let matrix: CooMat<f64> = CooMat::new(1, 1);
        assert_eq!(matrix.rows(), 1);
        let matrix: CooMat<f64> = CooMat::new(123, 1);
        assert_eq!(matrix.rows(), 123);
    }

    #[test]
    fn cols() {
        let matrix: CooMat<f64> = CooMat::new(1, 1);
        assert_eq!(matrix.cols(), 1);
        let matrix: CooMat<f64> = CooMat::new(1, 123);
        assert_eq!(matrix.cols(), 123);
    }

    #[test]
    fn shape() {
        let matrix: CooMat<f64> = CooMat::new(1, 1);
        assert_eq!(matrix.shape(), (1, 1));
        let matrix: CooMat<f64> = CooMat::new(123, 123);
        assert_eq!(matrix.shape(), (123, 123));
    }

    #[test]
    fn capacity_new() {
        let matrix: CooMat<f64> = CooMat::new(1, 1);
        assert_eq!(matrix.capacity(), 0);
    }

    #[test]
    fn capacity_with_capacity() {
        let matrix: CooMat<f64> = CooMat::with_capacity(1, 1, 0);
        assert_eq!(matrix.capacity(), 0);
        let matrix: CooMat<f64> = CooMat::with_capacity(1, 1, 1);
        assert_eq!(matrix.capacity(), 1);
        let matrix: CooMat<f64> = CooMat::with_capacity(1, 1, 123);
        assert_eq!(matrix.capacity(), 123);
    }

    #[test]
    fn capacity_with_entries() {
        let entries = vec![];
        let matrix: CooMat<f64> = CooMat::with_entries(1, 1, entries);
        assert_eq!(matrix.capacity(), 0);
        let entries = vec![(0, 0, 1.0)];
        let matrix = CooMat::with_entries(1, 1, entries);
        assert_eq!(matrix.capacity(), 1);
    }

    #[test]
    fn capacity_with_capacity_and_entries() {
        let entries = vec![];
        let matrix: CooMat<f64> = CooMat::with_capacity_and_entries(1, 1, 1, entries);
        assert_eq!(matrix.capacity(), 1);
    }

    #[test]
    fn capacity_with_triplets() {
        let rowind = vec![];
        let colind = vec![];
        let values: Vec<f64> = vec![];
        let matrix = CooMat::with_triplets(1, 1, rowind, colind, values);
        assert_eq!(matrix.capacity(), 0);
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![1.0];
        let matrix = CooMat::with_triplets(1, 1, rowind, colind, values);
        assert_eq!(matrix.capacity(), 1);
    }

    #[test]
    fn capacity_with_capacity_and_triplets() {
        let rowind = vec![];
        let colind = vec![];
        let values: Vec<f64> = vec![];
        let matrix = CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
        assert_eq!(matrix.capacity(), 1);
    }

    #[test]
    fn reserve() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: Vec::new(),
        };
        assert_eq!(matrix.capacity(), 0);
        matrix.reserve(10);
        assert!(matrix.entries.capacity() >= 10);
    }

    #[test]
    fn shrink() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: Vec::with_capacity(10),
        };
        assert_eq!(matrix.capacity(), 10);
        matrix.shrink();
        assert_eq!(matrix.capacity(), 0);
    }

    #[test]
    fn length_new() {
        let matrix: CooMat<f64> = CooMat::new(1, 1);
        assert_eq!(matrix.len(), 0);
    }

    #[test]
    fn length_with_capacity() {
        let matrix: CooMat<f64> = CooMat::with_capacity(1, 1, 1);
        assert_eq!(matrix.len(), 0);
    }

    #[test]
    fn length_with_entries() {
        let entries = vec![];
        let matrix: CooMat<f64> = CooMat::with_entries(1, 1, entries);
        assert_eq!(matrix.len(), 0);
        let entries = vec![(0, 0, 1.0)];
        let matrix = CooMat::with_entries(1, 1, entries);
        assert_eq!(matrix.len(), 1);
    }

    #[test]
    fn length_with_capcity_and_entries() {
        let entries = vec![];
        let matrix: CooMat<f64> = CooMat::with_capacity_and_entries(1, 1, 1, entries);
        assert_eq!(matrix.len(), 0);
    }

    #[test]
    fn length_with_triplets() {
        let rowind = vec![];
        let colind = vec![];
        let values: Vec<f64> = vec![];
        let matrix = CooMat::with_triplets(1, 1, rowind, colind, values);
        assert_eq!(matrix.len(), 0);
        let rowind = vec![0];
        let colind = vec![0];
        let values = vec![1.0];
        let matrix = CooMat::with_triplets(1, 1, rowind, colind, values);
        assert_eq!(matrix.len(), 1);
    }

    #[test]
    fn length_with_capcity_and_triplets() {
        let rowind = vec![];
        let colind = vec![];
        let values: Vec<f64> = vec![];
        let matrix = CooMat::with_capacity_and_triplets(1, 1, 1, rowind, colind, values);
        assert_eq!(matrix.len(), 0);
    }

    #[test]
    fn is_empty() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: Vec::new(),
        };
        assert!(matrix.is_empty());
        matrix.push(0, 0, 1.0);
        assert!(!matrix.is_empty());
    }

    #[test]
    fn truncate() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        };
        assert_eq!(matrix.len(), 4);
        matrix.truncate(2);
        assert_eq!(matrix.len(), 2);
    }

    #[test]
    fn truncate_clear() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        };
        assert_eq!(matrix.len(), 4);
        matrix.truncate(0);
        assert!(matrix.is_empty());
    }

    #[test]
    fn truncate_noop() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        };
        assert_eq!(matrix.len(), 4);
        matrix.truncate(4);
        assert_eq!(matrix.len(), 4);
    }

    #[test]
    #[should_panic]
    fn truncate_outofbounds() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        };
        matrix.truncate(5);
    }

    #[test]
    fn get() {
        let matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: vec![(0, 0, 1.0)],
        };
        assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
        assert_eq!(matrix.get(1), None);
    }

    #[test]
    fn get_mut() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: vec![(0, 0, 1.0)],
        };
        assert_eq!(matrix.get_mut(0), Some((&0, &0, &mut 1.0)));
        assert_eq!(matrix.get_mut(1), None);
    }

    #[test]
    fn insert() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (1, 1, 4.0)],
        };
        matrix.insert(1, 0, 1, 2.0);
        matrix.insert(2, 1, 0, 3.0);
        assert_eq!(matrix.entries.len(), 4);
        assert_eq!(matrix.entries.get(0), Some(&(0, 0, 1.0)));
        assert_eq!(matrix.entries.get(1), Some(&(0, 1, 2.0)));
        assert_eq!(matrix.entries.get(2), Some(&(1, 0, 3.0)));
        assert_eq!(matrix.entries.get(3), Some(&(1, 1, 4.0)));
    }

    #[test]
    #[should_panic]
    fn insert_outofbounds() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: vec![],
        };
        matrix.insert(1, 0, 0, 1.0);
    }

    #[test]
    fn remove() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: vec![(0, 0, 1.0)],
        };
        assert_eq!(matrix.entries.len(), 1);
        assert_eq!(matrix.remove(0), (0, 0, 1.0));
        assert!(matrix.entries.is_empty());
    }

    #[test]
    #[should_panic]
    fn remove_outofbounds() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: vec![],
        };
        matrix.remove(0);
    }

    #[test]
    fn push() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: vec![],
        };
        matrix.push(0, 0, 1.0);
        assert_eq!(matrix.entries.len(), 1);
        assert_eq!(matrix.entries.get(0), Some(&(0, 0, 1.0)));
    }

    #[test]
    fn pop() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: vec![(0, 0, 1.0)],
        };
        assert_eq!(matrix.pop(), Some((0, 0, 1.0)));
        assert!(matrix.entries.is_empty())
    }

    #[test]
    fn clear() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 1,
            cols: 1,
            entries: vec![(0, 0, 1.0)],
        };
        matrix.clear();
        assert!(matrix.entries.is_empty())
    }

    #[test]
    fn iter() {
        let matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        };
        let mut iter = matrix.iter();
        assert_eq!(iter.next(), Some((&0, &0, &1.0)));
        assert_eq!(iter.next(), Some((&0, &1, &2.0)));
        assert_eq!(iter.next(), Some((&1, &0, &3.0)));
        assert_eq!(iter.next(), Some((&1, &1, &4.0)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_mut() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        };
        let mut iter = matrix.iter_mut();
        assert_eq!(iter.next(), Some((&0, &0, &mut 1.0)));
        assert_eq!(iter.next(), Some((&0, &1, &mut 2.0)));
        assert_eq!(iter.next(), Some((&1, &0, &mut 3.0)));
        assert_eq!(iter.next(), Some((&1, &1, &mut 4.0)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn extend() {
        let mut matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![],
        };
        assert!(matrix.entries.is_empty());
        matrix.extend(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)]);
        assert_eq!(matrix.entries.len(), 4);
        assert_eq!(matrix.entries.get(0), Some(&(0, 0, 1.0)));
        assert_eq!(matrix.entries.get(1), Some(&(0, 1, 2.0)));
        assert_eq!(matrix.entries.get(2), Some(&(1, 0, 3.0)));
        assert_eq!(matrix.entries.get(3), Some(&(1, 1, 4.0)));
    }

    #[test]
    fn into_iter() {
        let matrix: CooMat<f64> = CooMat {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        };
        let mut iter = matrix.into_iter();
        assert_eq!(iter.next(), Some((0, 0, 1.0)));
        assert_eq!(iter.next(), Some((0, 1, 2.0)));
        assert_eq!(iter.next(), Some((1, 0, 3.0)));
        assert_eq!(iter.next(), Some((1, 1, 4.0)));
        assert!(iter.next().is_none());
    }
}
