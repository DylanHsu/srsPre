Sub ConvertDoc()
    Dim strFolder As String
    Dim strFile As String
    Dim FileSystem As Object
    Dim HostFolder As String
    With Application.FileDialog(4) ' msoFileDialogFolderPicker
        If .Show Then
            strFolder = .SelectedItems(1)
        Else
            MsgBox "No folder specified.", vbExclamation
            Exit Sub
        End If
    End With
    If Right(strFolder, 1) <> "\" Then
        strFolder = strFolder & "\"
    End If
    Application.ScreenUpdating = False
    Set FileSystem = CreateObject("Scripting.FileSystemObject")
    DoFolder FileSystem.GetFolder(strFolder)
    'strFile = Dir(strFolder & "*.doc")
    'Do While strFile <> ""
    '    ' Dir includes *.docx and *.docm with *.doc
    '    If LCase(Right(strFile, 4)) = ".doc" Then
    '        Set doc = Documents.Open(strFolder & strFile)
    '        If doc.HasVBProject Then
    '            doc.SaveAs FileName:=strFolder & strFile & "m", _
    '                FileFormat:=wdFormatXMLDocumentMacroEnabled
    '        Else
    '            doc.SaveAs FileName:=strFolder & strFile & "x", _
    '                FileFormat:=wdFormatXMLDocument
    '        End If
    '        doc.Close SaveChanges:=False
    '    End If
    '    strFile = Dir
    'Loop
    Application.ScreenUpdating = True
End Sub

Sub DoFolder(Folder)
    Dim doc As Document
    Dim SubFolder
    For Each SubFolder In Folder.SubFolders
        DoFolder SubFolder
    Next
    Dim File
    For Each File In Folder.Files
        ' Operate on each file
        If LCase(Right(File.Name, 4)) = ".doc" Then
            Set doc = Documents.Open(File.Path)
            'If doc.HasVBProject Then
            '    doc.SaveAs FileName:=File.Path & "m", _
            '        FileFormat:=wdFormatXMLDocumentMacroEnabled
            'Else
            '    doc.SaveAs FileName:=File.Path & "x", _
            '        FileFormat:=wdFormatXMLDocument
            'End If
            doc.SaveAs FileName:=File.Path & "x", _
                FileFormat:=wdFormatXMLDocument
            doc.Close SaveChanges:=False
        End If
    Next
End Sub