from distutils.core import setup

setup(name='pytorch_bn_fusion',
      version='1.0',
      description='Batch normalization fusion for PyTorch',
      author='Aleksei Tiulpin',
      author_email='lext@ods.ai',
      maintainer='Alexander Radionov',
      maintainer_email='alex.radionov@gmail.com',
      url='https://github.com/AlexanderRadionov/pytorch_bn_fusion',
      packages=['pytorch_bn_fusion'],
      package_data={'pytorch_bn_fusion': ['*.py']}
     )
