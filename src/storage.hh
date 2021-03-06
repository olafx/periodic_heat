#pragma once
#include <array>
#include <vtkNew.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkXMLImageDataWriter.h>

void store(void *const data, const std::array<std::size_t, 3> n, const char *const filename)
{
    vtkNew<vtkImageImport> image_import;
    image_import->SetDataSpacing(1, 1, 1);
    image_import->SetDataOrigin(0, 0, 0);
    image_import->SetWholeExtent(0, n[0] - 1, 0, n[1] - 1, 0, n[2] - 1);
    image_import->SetDataExtentToWholeExtent();
    image_import->SetDataScalarType(VTK_DOUBLE);
    image_import->SetNumberOfScalarComponents(2);
    image_import->SetImportVoidPointer(data);

    vtkNew<vtkXMLImageDataWriter> writer;
    writer->SetFileName(filename);
    writer->SetInputConnection(image_import->GetOutputPort());
    writer->Write();
}
