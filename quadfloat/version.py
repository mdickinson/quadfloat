"""
Version information.

"""
major = 0
minor = 1
patch = 0
prerelease = 'alpha'  # '', 'alpha', 'beta', etc.

if prerelease:
    __version__ = "{}.{}.{}-{}".format(major, minor, patch, prerelease)
else:
    __version__ = "{}.{}.{}".format(major, minor, patch)

# Release and version for Sphinx purposes.

# The short X.Y version.
version = "{}.{}".format(major, minor)

# The full version, including patchlevel and alpha/beta/rc tags.
release = __version__
