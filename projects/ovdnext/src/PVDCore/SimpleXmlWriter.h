/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2014 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Copyright (c) 2008-2014 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#ifndef SIMPLE_XML_WRITER_H
#define SIMPLE_XML_WRITER_H

#include "PxPvdFoundation.h"
#include <stdio.h>

namespace pvdapp
{
namespace repx
{

class XmlWriter
{
  public:
        struct STagWatcher
	{
                typedef XmlWriter TXmlWriterType;
                TXmlWriterType& mWriter;
                STagWatcher(const STagWatcher& inOther);
                STagWatcher& operator-(const STagWatcher& inOther);
                STagWatcher(TXmlWriterType& inWriter, const char* inTagName) : mWriter(inWriter)
		{
                        mWriter.beginTag(inTagName);
                }
                ~STagWatcher()
                {
                        mWriter.endTag();
                }

          private:
                // DJS: I added a dummy assignment operator to make MSVC happy
                STagWatcher& operator=(const STagWatcher&)
                {
                }
	};

        virtual ~XmlWriter()
	{
        }
        virtual void beginTag(const char* inTagname) = 0;
        virtual void endTag(bool useTabs = true) = 0;
        virtual void addAttribute(const char* inName, const char* inValue) = 0;
        virtual void writeContentTag(const char* inTag, const char* inContent) = 0;
        virtual void addContent(const char* inContent) = 0;
        virtual void writeTabs(physx::PxU32 inSize, bool useNewline = false) = 0;
};

template <typename TStreamType>
class XmlWriterImpl : public XmlWriter
{
        TStreamType& mStream;
        XmlWriterImpl(const XmlWriterImpl& inOther);
        XmlWriterImpl& operator=(const XmlWriterImpl& inOther);
        physx::shdfnd::Array<const char*> mTags;
        bool mTagOpen;
        physx::PxU32 mInitialTagDepth;

  public:
        XmlWriterImpl(TStreamType& inStream, physx::PxU32 inInitialTagDepth = 0)
        : mStream(inStream), mTags("char*"), mTagOpen(false), mInitialTagDepth(inInitialTagDepth)
        {
        }
        virtual ~XmlWriterImpl()
        {
                while(mTags.size())
                        endTag();
        }
        physx::PxU32 tabCount()
        {
                return static_cast<PxU32>(mTags.size()) + mInitialTagDepth;
        }

        void writeTabs(physx::PxU32 inSize, bool useNewline = false)
        {
                if(useNewline)
                        mStream << "\n";
                inSize += mInitialTagDepth;
                for(PxU32 idx = 0; idx < inSize; ++idx)
                        mStream << "\t";
        }
        void beginTag(const char* inTagname)
        {
                closeTag();
                writeTabs(static_cast<PxU32>(mTags.size()));
                mTags.pushBack(inTagname);
                mStream << "<" << inTagname;
                mTagOpen = true;
        }
        void addAttribute(const char* inName, const char* inValue)
        {
                PX_ASSERT(mTagOpen);
                mStream << " " << inName << "="
                        << "\"" << inValue << "\"";
        }
        void closeTag(bool useNewline = true)
        {
                if(mTagOpen)
		{
                        mStream << " "
                                << ">";
                        if(useNewline)
				mStream << "\n";
		}
                mTagOpen = false;
        }
        void doEndOpenTag()
        {
                mStream << "</" << mTags.back() << ">"
                        << "\n";
        }
        void endTag(bool useTabs = true)
        {
                PX_ASSERT(mTags.size());
                if(mTagOpen)
		{
                        mStream << " "
                                << "/>"
                                << "\n";
		}
                else
		{
                        if(useTabs)
                                writeTabs(static_cast<PxU32>(mTags.size()) - 1);
                        doEndOpenTag();
		}
                mTagOpen = false;
                mTags.popBack();
        }
        static bool IsNormalizableWhitespace(char c)
        {
                return c == 0x9 || c == 0xA || c == 0xD;
        }
        static bool IsValidXmlCharacter(char c)
        {
                return IsNormalizableWhitespace(c) || c >= 0x20;
        }

        void addContent(const char* inContent)
        {
                closeTag(false);

                // escape xml
                for(; *inContent; inContent++)
		{
                        switch(*inContent)
			{
                        case '<':
                                mStream << "&lt;";
                                break;
                        case '>':
                                mStream << "&gt;";
                                break;
                        case '&':
                                mStream << "&amp;";
                                break;
                        case '\'':
                                mStream << "&apos;";
                                break;
                        case '"':
                                mStream << "&quot;";
                                break;
                        default:
                                if(IsValidXmlCharacter(*inContent))
				{
                                        if(IsNormalizableWhitespace(*inContent))
                                        {
                                                char s[32];
                                                sprintf(s, "&#x%02X;", unsigned(*inContent));
                                                mStream << s;
					}
                                        else
                                                mStream << *inContent;
				}
                                break;
                        }
		}
        }
        void writeContentTag(const char* inTag, const char* inContent)
	{
                beginTag(inTag);
                addContent(inContent);
                doEndOpenTag();
                mTags.popBack();
        }
        void insertXml(const char* inXml)
	{
                closeTag();
                mStream << inXml;
        }
};

struct BeginTag
{
        const char* mTagName;
        BeginTag(const char* inTagName) : mTagName(inTagName)
	{
        }
};

struct EndTag
{
        EndTag()
	{
        }
};

struct Att
{
        const char* mAttName;
        const char* mAttValue;
        Att(const char* inAttName, const char* inAttValue) : mAttName(inAttName), mAttValue(inAttValue)
	{
        }
};

struct Content
{
        const char* mContent;
        Content(const char* inContent) : mContent(inContent)
	{
	}
};

struct ContentTag
{
        const char* mTagName;
        const char* mContent;
        ContentTag(const char* inTagName, const char* inContent) : mTagName(inTagName), mContent(inContent)
	{
	}
};

inline XmlWriter& operator<<(XmlWriter& inWriter, const BeginTag& inTag)
{
        inWriter.beginTag(inTag.mTagName);
        return inWriter;
}
inline XmlWriter& operator<<(XmlWriter& inWriter, const EndTag& /*inTag*/)
{
        inWriter.endTag();
        return inWriter;
}
inline XmlWriter& operator<<(XmlWriter& inWriter, const Att& inTag)
{
        inWriter.addAttribute(inTag.mAttName, inTag.mAttValue);
        return inWriter;
}
inline XmlWriter& operator<<(XmlWriter& inWriter, const Content& inTag)
{
        inWriter.addContent(inTag.mContent);
        return inWriter;
}
inline XmlWriter& operator<<(XmlWriter& inWriter, const ContentTag& inTag)
{
        inWriter.writeContentTag(inTag.mTagName, inTag.mContent);
        return inWriter;
}

inline void writeProperty(XmlWriter& inWriter, physx::pvdsdk::ForwardingMemoryBuffer& tempBuffer, const char* inPropName)
{
        inWriter.writeContentTag(inPropName, tempBuffer.cStr()); // reinterpret_cast<const char*>(tempBuffer.begin())
        tempBuffer.clear();
}
template <typename TDataType>
inline void writeProperty(XmlWriter& inWriter, physx::pvdsdk::ForwardingMemoryBuffer& tempBuffer,
                          const char* inPropName, TDataType inValue)
{
        tempBuffer << inValue;
        writeProperty(inWriter, tempBuffer, inPropName);
}
}
}

#endif
